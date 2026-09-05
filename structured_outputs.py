from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    ModelSettings,
    set_tracing_disabled,
)
import asyncio
import os
import sys
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field

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

# Use qwen/qwen3.8-27b on Groq which has first-class Structured Outputs (json_schema) support
model = OpenAIChatCompletionsModel(
    model="qwen/qwen3.8-27b",
    openai_client=client,
)

# -------------------------------------------------------------------------
# 1. Define Pydantic Models for Structured Output
# -------------------------------------------------------------------------
class OrderItem(BaseModel):
    item_name: str = Field(description="Name of the food or item")
    quantity: int = Field(description="Number of units ordered")
    unit_price: float = Field(description="Price per single unit in USD")

class CustomerOrder(BaseModel):
    customer_name: str = Field(description="Name of the customer placing order")
    items: list[OrderItem] = Field(description="List of items ordered")
    delivery_address: str = Field(description="Delivery destination address")
    is_rush_delivery: bool = Field(description="True if customer requested express/rush delivery")
    estimated_delivery_minutes: int = Field(description="Estimated delivery time in minutes")

# -------------------------------------------------------------------------
# 2. Create Agent with output_type & max_tokens limit for Groq OTPM
# -------------------------------------------------------------------------
order_agent = Agent(
    name="Order Extractor Agent",
    instructions="You extract structured restaurant delivery orders from unstructured user messages.",
    output_type=CustomerOrder,  # <-- Enforces strict Pydantic return type!
    model=model,
    model_settings=ModelSettings(max_tokens=400),
)

# -------------------------------------------------------------------------
# 3. Run and Consume Typed Output
# -------------------------------------------------------------------------
async def main():
    user_message = (
        "Hi, I am Sarah Connor. I'd like to order 2 Pepperoni Pizzas at $15.50 each "
        "and 3 Cokes at $2.50 each. Please deliver them as fast as possible to "
        "742 Evergreen Terrace. How long will it take?"
    )

    print("=" * 60)
    print("💬 Unstructured Input:")
    print(user_message)
    print("=" * 60)

    # Run the agent
    result = await Runner.run(order_agent, user_message)

    # result.final_output is automatically an instance of CustomerOrder!
    order: CustomerOrder = result.final_output

    print("\n📦 Extracted Structured Data (Typed Pydantic Object):")
    print(f"Customer Name   : {order.customer_name}")
    print(f"Delivery Address: {order.delivery_address}")
    print(f"Rush Delivery?  : {'Yes 🚀' if order.is_rush_delivery else 'No'}")
    print(f"Estimated Time  : {order.estimated_delivery_minutes} mins")

    print("\n🛒 Items Ordered:")
    total = 0.0
    for item in order.items:
        subtotal = item.quantity * item.unit_price
        total += subtotal
        print(f"  • {item.item_name} x{item.quantity} @ ${item.unit_price:.2f} = ${subtotal:.2f}")

    print(f"\n💵 Total Bill: ${total:.2f}")

    print("\n" + "=" * 60)
    print("📄 Exported JSON (order.model_dump_json()):")
    print(order.model_dump_json(indent=2))
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
