from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    ModelSettings,
    RunContextWrapper,
    function_tool,
    handoff,
    set_tracing_disabled,
)
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import asyncio
import os
import sys
from dotenv import load_dotenv
import openai

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)

# Remove OPENAI_API_KEY so the Agents SDK doesn't attempt OpenAI tracing (prevents 429 errors)
os.environ.pop("OPENAI_API_KEY", None)
set_tracing_disabled(True)

# Setup Groq Async client
client = openai.AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

model = OpenAIChatCompletionsModel(
    model="qwen/qwen3.6-27b",
    openai_client=client,
)

def sanitize_schema_for_groq(schema_dict: dict):
    """Ensure JSON schemas adhere strictly to Groq API validation rules."""
    if not schema_dict.get("properties"):
        schema_dict.pop("required", None)
    else:
        schema_dict["required"] = list(schema_dict["properties"].keys())

# =========================================================================
# 1. GRAPH STATE (Shared Workflow Context across Graph Nodes)
# =========================================================================
@dataclass
class WorkflowState:
    """Shared state tracked across all nodes in the evaluation graph."""
    topic: str
    current_draft: str = ""
    feedback_notes: list[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3
    is_approved: bool = False

# =========================================================================
# 2. FINAL STRUCTURED OUTPUT SCHEMA (Graph Exit Contract)
# =========================================================================
class EvaluationReport(BaseModel):
    """The final approved payload produced at the graph exit node."""
    title: str = Field(description="Catchy title of the final content")
    final_content: str = Field(description="The finalized, polished content")
    quality_score: int = Field(description="Score from 1 to 100")
    total_iterations: int = Field(description="Number of draft-review cycles taken")
    review_summary: str = Field(description="Brief summary of why this draft was approved")

# =========================================================================
# 3. STATE MANAGEMENT TOOLS (Dependency Injected into Nodes)
# =========================================================================
@function_tool
def save_draft(ctx: RunContextWrapper[WorkflowState], draft_text: str) -> str:
    """Save the newly generated or revised draft into the shared graph state."""
    state = ctx.context
    state.iteration += 1
    state.current_draft = draft_text
    return f"Draft #{state.iteration} saved to graph state: {draft_text}"

@function_tool
def record_feedback(ctx: RunContextWrapper[WorkflowState], feedback: str) -> str:
    """Record reviewer feedback into graph state to guide the next writing iteration."""
    state = ctx.context
    state.feedback_notes.append(feedback)
    return f"Feedback recorded: {feedback}"

# Sanitize tool schemas for Groq API
sanitize_schema_for_groq(save_draft.params_json_schema)
sanitize_schema_for_groq(record_feedback.params_json_schema)

# =========================================================================
# 4. GRAPH NODES (Agents) & EDGES (Handoffs)
# =========================================================================

# Node A: Writer Node (Optimizer)
writer_node = Agent(
    name="Writer Node",
    instructions=(
        "You are the Writer Node in an Evaluator-Optimizer graph.\n"
        "Task: Generate or refine a concise 2-sentence draft based on the topic and any reviewer feedback.\n"
        "Step 1: Call `save_draft` with the draft text.\n"
        "Step 2: Immediately hand off to Reviewer Node."
    ),
    tools=[save_draft],
    model=model,
    model_settings=ModelSettings(max_tokens=350),
)

# Node C: Publisher Node (Exit / Formatter Node with Structured Output)
publisher_node = Agent(
    name="Publisher Node",
    instructions=(
        "You are the Publisher Node. Review the latest draft from the workflow and produce "
        "the final EvaluationReport structured output."
    ),
    output_type=EvaluationReport,
    model=model,
    model_settings=ModelSettings(max_tokens=400),
)

# Node B: Reviewer Node (Evaluator / Decision Gate)
reviewer_node = Agent(
    name="Reviewer Node",
    instructions=(
        "You are the Reviewer / Evaluator Node.\n"
        "Evaluate the draft in context (score 1-100 for clarity, punchiness, and constraints).\n"
        "- If score >= 85 or if a revision was already made: Hand off immediately to Publisher Node.\n"
        "- If score < 85: First call `record_feedback` with specific improvement advice, then hand off to Writer Node."
    ),
    tools=[record_feedback],
    model=model,
    model_settings=ModelSettings(max_tokens=350),
)

# Define Graph Edges (Transitions)
to_reviewer = handoff(reviewer_node)
sanitize_schema_for_groq(to_reviewer.input_json_schema)

to_writer = handoff(writer_node)
sanitize_schema_for_groq(to_writer.input_json_schema)

to_publisher = handoff(publisher_node)
sanitize_schema_for_groq(to_publisher.input_json_schema)

# Wire the Graph Topology:
# Writer -> Reviewer
writer_node.handoffs = [to_reviewer]
# Reviewer -> Writer (Loop) OR Reviewer -> Publisher (Exit)
reviewer_node.handoffs = [to_writer, to_publisher]

# =========================================================================
# 5. GRAPH EXECUTION WITH REAL-TIME STREAMING
# =========================================================================
async def run_evaluator_optimizer_graph(topic_prompt: str):
    print("=" * 70)
    print("🚀 STARTING EVALUATOR-OPTIMIZER AGENT GRAPH")
    print(f"📌 Goal: {topic_prompt}")
    print("=" * 70)

    state = WorkflowState(topic=topic_prompt)

    # Execute graph starting at Writer Node with turn limit raised
    result = Runner.run_streamed(
        starting_agent=writer_node,
        input=f"Goal: {topic_prompt}",
        context=state,
        max_turns=25,
    )

    current_node = "Writer Node"
    print(f"\n🟢 [Node: {current_node}] Activated")

    async for event in result.stream_events():
        # A. Graph Edge Transitions
        if event.type == "agent_updated_stream_event":
            current_node = event.new_agent.name
            print(f"\n\n🔀 [Graph Edge] Transitioning -> 🟢 {current_node}")

        # B. Node Tool Actions
        elif event.type == "run_item_stream_event":
            if event.name == "tool_called":
                tool_name = getattr(event.item, "tool_name", "tool")
                print(f"\n  ⚙️  [{current_node}] Calling: {tool_name}...")
            elif event.name == "tool_output":
                output = getattr(event.item, "output", "")
                print(f"  ✅ [{current_node}] Result: {output}")

        # C. Token Streaming
        elif event.type == "raw_response_event":
            if hasattr(event.data, "delta") and event.data.delta:
                print(event.data.delta, end="", flush=True)

    print("\n\n" + "=" * 70)
    print("🏁 GRAPH WORKFLOW COMPLETED")
    print("=" * 70)

    # Display final structured report & internal graph state
    report: EvaluationReport = result.final_output
    if report and isinstance(report, EvaluationReport):
        print(f"\n📊 Final Evaluator Report (Structured Pydantic Data):")
        print(f"  • Title             : {report.title}")
        print(f"  • Quality Score     : {report.quality_score}/100")
        print(f"  • Total Iterations  : {report.total_iterations}")
        print(f"  • Reviewer Summary  : {report.review_summary}")
        print(f"\n📝 Final Approved Copy:\n{report.final_content}")
    else:
        print(f"Final output: {report}")

    print("\n" + "-" * 70)
    print("🏦 Final Python Graph State:")
    print(f"  • Total Draft Cycles: {state.iteration}")
    print(f"  • Feedback History  : {len(state.feedback_notes)} items")
    for idx, fb in enumerate(state.feedback_notes, 1):
        print(f"    {idx}. {fb}")
    print("-" * 70)

async def main():
    task = "Write a compelling 2-sentence launch announcement for an open-source AI agent framework."
    await run_evaluator_optimizer_graph(task)

if __name__ == "__main__":
    asyncio.run(main())
