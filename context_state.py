from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    ModelSettings,
    RunContextWrapper,
    function_tool,
    set_tracing_disabled,
)
from agents.memory.sqlite_session import SQLiteSession
from dataclasses import dataclass, field
import asyncio
import os
import sys
from dotenv import load_dotenv
import openai

# Ensure Unicode output works on Windows terminals
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
    model="qwen/qwen3.8-27b",
    openai_client=client,
)

# -------------------------------------------------------------------------
# 1. Define Custom Application State / Context (Dependency Injection)
# -------------------------------------------------------------------------
@dataclass
class UserAccount:
    """Application state passed into tools at runtime without exposing to the LLM."""
    user_id: str
    name: str
    balance: float
    cart: list[str] = field(default_factory=list)

# -------------------------------------------------------------------------
# 2. Define Tools with Injected RunContextWrapper
# -------------------------------------------------------------------------
@function_tool
def get_account_details(ctx: RunContextWrapper[UserAccount]) -> str:
    """Retrieve the current user account ID, name, balance, and cart items."""
    user = ctx.context  # Access the injected Python object
    cart_items = ", ".join(user.cart) if user.cart else "Empty"
    return f"User: {user.name} (ID: {user.user_id}) | Balance: ${user.balance:.2f} | Cart: [{cart_items}]"

# Groq schema patch for 0-arg function tool
get_account_details.params_json_schema.pop("required", None)

@function_tool
def purchase_item(ctx: RunContextWrapper[UserAccount], item_name: str, price: float) -> str:
    """Purchase an item, deduct its price from balance, and add it to the cart."""
    user = ctx.context  # Modify the shared Python state
    if user.balance < price:
        return f"Transaction declined: Insufficient funds. Balance is ${user.balance:.2f}, but {item_name} costs ${price:.2f}."
    
    user.balance -= price
    user.cart.append(item_name)
    return f"Success: Purchased '{item_name}' for ${price:.2f}. New balance is ${user.balance:.2f}."

# -------------------------------------------------------------------------
# 3. Define the Agent
# -------------------------------------------------------------------------
banking_agent = Agent(
    name="Banking Assistant",
    instructions="You are a personal shopping and banking assistant. Always use tools to check account info or perform purchases.",
    tools=[get_account_details, purchase_item],
    model=model,
    model_settings=ModelSettings(max_tokens=350),
)

# -------------------------------------------------------------------------
# 4. Run Multi-Turn Conversation with Context & Session
# -------------------------------------------------------------------------
async def main():
    # A. Create the application state
    current_user = UserAccount(user_id="ACC-8492", name="Alex Mercer", balance=150.0)

    # B. Create persistent conversation session (SQLite in-memory or on-disk)
    session = SQLiteSession(session_id="alex_session_001", db_path=":memory:")

    print("=" * 65)
    print("🏦 Initial Application State in Python:")
    print(f"  User ID : {current_user.user_id}")
    print(f"  Name    : {current_user.name}")
    print(f"  Balance : ${current_user.balance:.2f}")
    print(f"  Cart    : {current_user.cart}")
    print("=" * 65)

    # --- Turn 1: Inspect state via tool ---
    prompt1 = "Hi! Can you check what my account balance is?"
    print(f"\n💬 User (Turn 1): {prompt1}")
    res1 = await Runner.run(banking_agent, prompt1, context=current_user, session=session)
    print(f"🤖 Agent: {res1.final_output}\n")

    # --- Turn 2: Mutate state via tool ---
    prompt2 = "I want to buy a Mechanical Keyboard for $85.00."
    print(f"💬 User (Turn 2): {prompt2}")
    res2 = await Runner.run(banking_agent, prompt2, context=current_user, session=session)
    print(f"🤖 Agent: {res2.final_output}\n")

    # --- Turn 3: Test multi-turn conversational memory ---
    prompt3 = "What did I just buy and how much money do I have left?"
    print(f"💬 User (Turn 3): {prompt3}")
    res3 = await Runner.run(banking_agent, prompt3, context=current_user, session=session)
    print(f"🤖 Agent: {res3.final_output}\n")

    print("=" * 65)
    print("🏦 Final Application State in Python (Mutated by Agent):")
    print(f"  User ID : {current_user.user_id}")
    print(f"  Name    : {current_user.name}")
    print(f"  Balance : ${current_user.balance:.2f}")
    print(f"  Cart    : {current_user.cart}")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(main())
