# Given an order, determine whether it should be refunded or not. 

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

orders = {
    "12345": {
        "order_id": "12345",
        "purchase_date": "01-07-2026",
        "refund_status": "not_refunded",
        "payment_method": "credit_card"
    }
}

class OrderState(TypedDict):
    order_id: str
    purchase_date: str
    refund_status: str
    payment_method: str
    eligible_for_refund: bool
    rejection_reason: str

def load_order_node(state: OrderState) -> OrderState:
    if state["order_id"] == "":
        return {
            "order_id": "12345",
            "purchase_date": orders["12345"].get("purchase_date"),
            "refund_status": orders["12345"].get("refund_status"),
            "payment_method": orders["12345"].get("payment_method")
        }
    else: 
        order_id = state.get("order_id")
        return {
            "purchase_date": orders[order_id].get("purchase_date"),
            "refund_status": orders[order_id].get("refund_status"),
            "payment_method": orders[order_id].get("payment_method")
        }

model = init_chat_model("gpt-5-nano")

def check_eligibility_node(state: OrderState) -> OrderState:

    response = model.invoke(f"""
        You are evaluating whether an order is eligible for a refund.
        Company policy:
        Refunds are allowed within 30 days of purchase.
        Purchase date: {state["purchase_date"]} (dd-mm-yy)
        Return only "true" or "false" as a string.
        "true" - model is eligible for refund.
        "false" - model is not eligible for refund.                          
    """)
    print("model response,", response.text)
    if response.text == "true":
        return { 
            "eligible_for_refund": True
        }
    elif response.text == "false":
        return { 
            "eligible_for_refund": False
        }

def refund_node(state: OrderState) -> OrderState:  
    agent = create_agent(
        model="openai:gpt-5-nano",
        system_prompt="""You are a helpful refund assistant.
        Your role is to refund a customer's order when provided with an order id.""",
        tools=[refund_order]
    )

    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Process refund for order id {state["order_id"]}",
                }
            ],
        },
    )

    return {
        "refund_status": "refunded"
    }

@tool
def refund_order(order_id: str) -> bool:
    """Use this tool to refund orders.
    Args: order_id - refers to the order id by which to look up an order.
    """
    print("refunding order")
    order = orders[order_id]
    order["refund_status"] = "refunded"
    return True 

def reject_node(state: OrderState) -> OrderState: 
    return { 
        "rejection_reason": "Purchase date exceeds refund eligibility period."
    }

def conditional_edge(state: OrderState) -> Literal["refund_node", "reject_node"]:
    if state["eligible_for_refund"] == True:
        return "refund_node"
    else:
        return "reject_node"

builder = StateGraph(OrderState)

builder.add_node("load_order_node", load_order_node)
builder.add_node("check_eligibility_node", check_eligibility_node)
builder.add_node("refund_node", refund_node)
builder.add_node("reject_node", reject_node)

builder.add_edge(START, "load_order_node")
builder.add_edge("load_order_node", "check_eligibility_node")

builder.add_conditional_edges("check_eligibility_node", conditional_edge)

builder.add_edge("refund_node", END)
builder.add_edge("reject_node", END)

graph = builder.compile()

result = graph.invoke({"order_id": "12345"})

print(result)