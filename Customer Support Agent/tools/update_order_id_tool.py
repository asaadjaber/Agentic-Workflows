from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from customer_information_state import CustomerInformationState
from langgraph.types import Command
from database import orders

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