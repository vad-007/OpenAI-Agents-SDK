import json
from dotenv import load_dotenv
import openai
import asyncio
import os
from agents import (
    Agent,
    Runner,
    ModelSettings,
    OpenAIChatCompletionsModel,
    ToolGuardrailFunctionOutput,
    function_tool,
    tool_input_guardrail,
    tool_output_guardrail,
    set_tracing_disabled,
)


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


@tool_input_guardrail
def block_secrets(data):
    args = json.loads(data.context.tool_arguments or "{}")
    if "sk-" in json.dumps(args):
        return ToolGuardrailFunctionOutput.reject_content(
            "Remove secrets before calling this tool."
        )
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail
def redact_output(data):
    text = str(data.output or "")
    if "sk-" in text:
        return ToolGuardrailFunctionOutput.reject_content("Output contained sensitive data.")
    return ToolGuardrailFunctionOutput.allow()


@function_tool(
    tool_input_guardrails=[block_secrets],
    tool_output_guardrails=[redact_output],
)
def classify_text(text: str) -> str:
    """Classify text for internal routing."""
    return f"length:{len(text)}"


agent = Agent(
    name="Classifier", 
    model=model,
    model_settings=groq_settings,
    tools=[classify_text],
)

async def main():
    result = await Runner.run(agent, "Please classify the text 'hello world' using the classify_text tool.")
    print("Agent output:", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())