from agents import Agent, OpenAIChatCompletionsModel, handoff, run_demo_loop, set_tracing_disabled
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

# Specialist Agent 1: History
history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You answer history questions clearly and concisely.",
    model=model,
)

# Specialist Agent 2: Math
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You explain math step by step and include worked examples only those Question where user ask explain other wise just be concise or short",
    model=model,
)

# Configure handoffs for Groq schema compatibility
history_handoff = handoff(history_tutor_agent)
history_handoff.input_json_schema.pop("required", None)

math_handoff = handoff(math_tutor_agent)
math_handoff.input_json_schema.pop("required", None)

# Main Triage Agent
triage_agent = Agent(
    name="Triage Agent",
    instructions="Route each homework question to the right specialist. If the question is not about math or history, answer it directly yourself.",
    handoffs=[history_handoff, math_handoff],
    model=model,
)

async def main():
    print("--- Starting REPL Utility (Interactive Chat Loop) ---")
    print("Type your questions below. Enter 'exit' or 'quit' to stop.")
    await run_demo_loop(triage_agent)

if __name__ == "__main__":
    asyncio.run(main())
