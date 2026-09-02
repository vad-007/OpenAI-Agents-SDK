from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, handoff, run_demo_loop, set_tracing_disabled
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

# Setup async client for Groq
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

# Specialist Agent 1: History
history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You answer history questions. CRITICAL: Your response must be 30 words or less. Do not exceed this limit unless the user explicitly asks for more details or a list of books/resources.",
    model=model,
    model_settings=groq_settings,
)

# Specialist Agent 2: Math
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You explain math. CRITICAL: Your response must be 30 words or less, unless the user explicitly asks for a step-by-step explanation or worked examples.",
    model=model,
    model_settings=groq_settings,
)

# Configure handoffs for Groq schema compatibility
history_handoff = handoff(history_tutor_agent)
history_handoff.input_json_schema.pop("required", None)

math_handoff = handoff(math_tutor_agent)
math_handoff.input_json_schema.pop("required", None)

# Main Triage Agent
triage_agent = Agent(
    name="Triage Agent",
    instructions="Route each homework question to the right specialist. If the question is not about math or history, answer it directly yourself. CRITICAL: When answering directly, your response must be 30 words or less.",
    handoffs=[history_handoff, math_handoff],
    model=model,
    model_settings=groq_settings,
)

async def main():
    print("--- Starting REPL Utility (Interactive Chat Loop) ---")
    print("Type your questions below. Enter 'exit' or 'quit' to stop.")
    await run_demo_loop(triage_agent)

if __name__ == "__main__":
    asyncio.run(main())
