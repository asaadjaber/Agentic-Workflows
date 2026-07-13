from database import orders
from langchain.tools import tool

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