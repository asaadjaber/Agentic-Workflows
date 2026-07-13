from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

orders = {
    "12345": {
        "order_id": "12345",
        "purchase_date": "01-07-2026",
        "refund_status": "not_refunded",
        "shipping_status": "delivered",
        "payment_method": "visa"
    }
}

customers = {
    "abc_123": { 
        "customer_id": "abc_123",
        "name": "Roy Farzan",
        "age": "38",
        "orders": {
            "12345": {
                "order_id": "12345",
                "purchase_date": "01-07-2026" 
            }
        }
    }
}

class CustomAgentState(AgentState): 
    customer_name: str
    customer_id: str
    order_id: str
    current_issue: str
    tracking_number: str

model = "openai:gpt-5-nano"

agent = create_agent(
    model=model,
    system_prompt="You are a helpful customer support agent. Help users with their orders in a concise and polite manner.",
    state_schema=CustomAgentState,
    tools=[],
    middleware=[],
    checkpointer=InMemorySaver()
)

thread_config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("You: ")

    if user_input.lower() == "finish":
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
        },
        thread_config,
    )

    print("Assistant:", result["messages"][-1].text)