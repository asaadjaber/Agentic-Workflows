from database import orders
from langchain.tools import tool

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