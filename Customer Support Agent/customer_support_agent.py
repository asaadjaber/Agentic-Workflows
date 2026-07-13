from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

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
        "first_name": "Roy",
        "last_name": "Farzan",
        "email": "royfarzan@abc.xyz",
        "age": "38",
        "orders": {
            "12345": {
                "order_id": "12345",
                "purchase_date": "01-07-2026" 
            }
        }
    }
}

class CustomerInformationState(AgentState): 
    first_name: str
    customer_id: str
    order_id: str
    current_issue: str
    tracking_number: str

@tool
def fetch_customer_information(email: str, runtime: ToolRuntime[None, CustomerInformationState] ) -> Command: 
    """This tool fetches a customer's information. Look up is by email.
        Arguments: email - the email of a user by which to look up customer info. 
    """
    for customer_key in customers:
        customer = customers[customer_key]
        if customer.get("email") == email:
            return Command(
                update={
                    "first_name": customer["first_name"],
                    "customer_id": customer["customer_id"],
                    "messages": [
                        ToolMessage(
                            content=f"Updated name to {customer["first_name"]}, and id to {customer["customer_id"]}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

model = "openai:gpt-5-nano"

agent = create_agent(
    model=model,
    system_prompt="""You are a helpful customer support agent. Help users with their orders in a concise and polite manner.
                     Begin by asking a customer for their customer email, and look up their customer information. Then address them by their first name.  
                  """,
    state_schema=CustomerInformationState,
    tools=[fetch_customer_information],
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