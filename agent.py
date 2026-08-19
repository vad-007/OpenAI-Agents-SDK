from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
import asyncio
import os
from dotenv import load_dotenv
import openai

load_dotenv(override=True)

# Disable tracing — it tries to upload traces to OpenAI's backend using
# OPENAI_API_KEY, which you don't have set for Groq
set_tracing_disabled(True)

# IMPORTANT: use AsyncOpenAI, not OpenAI — Runner.run is async
client = openai.AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",   # Groq's model id, kept as-is
    openai_client=client
)

agent = Agent(
    name="History Tutor",
    instructions="You answer history questions clearly and concisely.",
    model=model,
)

async def main():
    result = await Runner.run(agent, "When did the Roman Empire fall?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())