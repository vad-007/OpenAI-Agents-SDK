from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings, handoff, set_tracing_disabled
import asyncio
import os
import sys
from dotenv import load_dotenv
import openai

# Ensure Unicode output works on Windows terminals
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)

# Remove OPENAI_API_KEY so the Agents SDK doesn't attempt tracing via OpenAI (which causes 429 quota errors)
os.environ.pop("OPENAI_API_KEY", None)
set_tracing_disabled(True)

# IMPORTANT: use AsyncOpenAI, not OpenAI — Runner.run is async
client = openai.AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

model = OpenAIChatCompletionsModel(
    model="qwen/qwen3.6-27b",
    openai_client=client
)

# Suppress reasoning <think> traces in final response
groq_settings = ModelSettings(extra_body={"reasoning_format": "hidden"})

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You answer history questions. CRITICAL: Your response must be 30 words or less. Do not exceed this limit unless the user explicitly asks for more details or resources.",
    model=model,
    model_settings=groq_settings,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You explain math. CRITICAL: Your response must be 30 words or less, unless the user explicitly asks for a step-by-step explanation or worked examples.",
    model=model,
    model_settings=groq_settings,
)

# Groq rejects 'required' in function schemas when 'properties' is empty.
# Wrapping agents in handoff() and stripping 'required' ensures valid schemas for Groq.
history_handoff = handoff(history_tutor_agent)
history_handoff.input_json_schema.pop("required", None)

math_handoff = handoff(math_tutor_agent)
math_handoff.input_json_schema.pop("required", None)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route each homework question to the right specialist. If you answer directly, keep your response under 30 words.",
    handoffs=[history_handoff, math_handoff],
    model=model,
    model_settings=groq_settings,
)

async def main():
    result = await Runner.run(
        triage_agent,
        "Who was the first president of the United States?",
    )
    print(result.final_output)
    print(f"Answered by: {result.last_agent.name}")

if __name__ == "__main__":
    asyncio.run(main())