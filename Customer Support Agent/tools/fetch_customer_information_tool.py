from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

from customer_information_state import CustomerInformationState

from database import customers
from database import orders

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