from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
import asyncio
import os
import sys
from dotenv import load_dotenv
import openai

# Ensure Unicode output works on Windows terminals
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)
set_tracing_disabled(True)

client = openai.AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

model = OpenAIChatCompletionsModel(
    model="qwen/qwen3.6-27b",
    openai_client=client
)

# 1. Define your tool(s) with @function_tool
@function_tool
def history_fun_fact() -> str:
    """Return a short, surprising history or science fact."""
    return "Sharks are older than trees."

# Groq rejects an empty 'required' list when 'properties' is also empty.
# Strip it so the schema is valid for Groq's API.
history_fun_fact.params_json_schema.pop("required", None)

# 2. Register tools on the Agent
agent = Agent(
    name="History Tutor",
    instructions="You answer history questions clearly and concisely. Use tools when they help you give a precise answer.",
    model=model,
    tools=[history_fun_fact],   # <-- new
)

async def main():
    result = await Runner.run(agent, "Tell me something surprising about ancient life on Earth.")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())