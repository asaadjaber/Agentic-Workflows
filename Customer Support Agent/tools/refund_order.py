from langchain.tools import tool
from database import orders

@tool
def refund_order(order_id: str) -> bool: 
    """This tool is for refunding orders for customers. 
    Ensure checking whether purchase date falls within refund policy window before issuing refund. 
    Args: order_id - represents the order of the id of the order to be refunded.  
    """
    for order_key in orders:
        if order_key == order_id:
            order = orders[order_key]
            order["refund_status"] = "refunded"
            orders[order_key] = order
            return True

    return False
