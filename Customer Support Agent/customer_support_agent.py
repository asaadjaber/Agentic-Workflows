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
def fetch_customer_information(email: str, runtime: ToolRuntime[None, CustomerInformationState]) -> Command: 
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

@tool
def update_order_id(order_number: int, runtime: ToolRuntime[None, CustomerInformationState]) -> Command:
    """This tool updates the the order id in the customer information state when a customer provides an order number. 
        Arguments: order_number - the order number by which to look up an order for which to set in state. 
    """
    print("updating order_id in state")
    for order_key in orders:
        if order_key == order_number:
            order = orders[order_key]
            order_id = order.get("order_id")
            return Command(
                update={
                    "order_id": order_id,
                    "messages": [
                        ToolMessage(
                            content=f"Updated order id to {order_id}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

@tool 
def fetch_order_information(order_number: str) -> dict | None: 
    """This tool fetches the order by order number when a customer requests a lookup. 
        Use this tool to pull up order information. 
        Arguments: order_number - the order number by which to look up an order. 
    """
    print("fetching order information")
    for order_key in orders: 
        if order_key == order_number:
            order = orders[order_key]
            return order
        
    return None

@tool 
def check_shipping_status(order_number: str) -> str | None: 
    """This tool fetches the shipping status of the current order.
        Use this tool to pull up the shipping status directly. 
        Arguments: order_number: the order number for which to look up an order.
    """
    print("checking shipping status")
    for order_key in orders:
        if order_key == order_number:
            order = orders[order_key]
            shipping_status = order.get("shipping_status")
            return shipping_status

    return None

@tool 
def update_current_issue(str: str, runtime: ToolRuntime[None, CustomerInformationState]) -> Command: 
    """This tool updates the current issue in the customer information state when a new issue arises. 
        Use this to keep track of the current issue the customer is experiencing.
        Arguments: str - the issue which to update in the information state. 
    """
    print("updating current issue")
    return Command(
        update={
            "current_issue": str,
            "messages": [
                ToolMessage(
                    content=f"Updated current issue to {str}",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

model = "openai:gpt-5-nano"

agent = create_agent(
    model=model,
    system_prompt="""You are a helpful customer support agent. Help users with their orders in a concise and polite manner.
                     Begin by asking a customer for their customer email, and look up their customer information. 
                     Then address them by their first name.
                     When provided an order number, the first thing you should do is update the order id in the information state. 
                     You can afterwards look up the order by the order id.  
                     Update the current issue the customer is experiencing whenever it is appropriate to do so.  
                  """,
    state_schema=CustomerInformationState,
    tools=[fetch_customer_information, 
           update_order_id, fetch_order_information, 
           update_current_issue, 
           check_shipping_status],
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