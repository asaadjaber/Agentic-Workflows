# Given an order, determine whether it should be refunded or not. 

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

orders = {
    "12345": {
        "order_id": "12345",
        "purchase_date": "01-07-2026",
        "refund_status": "not_refunded",
        "payment_method": "credit_card",
        "order_amount": 90 
    }
}

class OrderState(TypedDict):
    order_id: str
    purchase_date: str
    refund_status: str
    payment_method: str
    eligible_for_refund: bool
    rejection_reason: str
    order_amount: int
    requires_human_approval: bool
    human_approved: bool

def load_order_node(state: OrderState) -> dict:
    if state["order_id"] == "":
        return {
            "order_id": "12345",
            "purchase_date": orders["12345"].get("purchase_date"),
            "refund_status": orders["12345"].get("refund_status"),
            "payment_method": orders["12345"].get("payment_method"),
            "order_amount": orders["12345"].get("order_amount")
        }
    else: 
        order_id = state.get("order_id")
        return {
            "purchase_date": orders[order_id].get("purchase_date"),
            "refund_status": orders[order_id].get("refund_status"),
            "payment_method": orders[order_id].get("payment_method"),
            "order_amount": orders[order_id].get("order_amount")
        }

model = init_chat_model("gpt-5-nano")

def check_eligibility_node(state: OrderState) -> dict:

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
    
def conditional_edge(state: OrderState) -> Literal["check_approval_requirement_node", "reject_node"]:
    if state["eligible_for_refund"] == True:
        return "check_approval_requirement_node"
    else:
        return "reject_node"
    
def check_approval_requirement_node(state: OrderState) -> dict:
    if state["order_amount"] >= 100: 
        return {
            "requires_human_approval": True
        }
    else: 
        return {
            "requires_human_approval": False
        }

def check_approval_requirement_router(state: OrderState) -> Literal["refund_node", "human_review_node"]:
    if state["requires_human_approval"] == True:
        return "human_review_node"
    else: 
        return "refund_node"

def human_review_node(state: OrderState) -> Command[Literal["refund_node", "reject_node"]]:

    decision = interrupt(
        {
            "question": "Approve this refund?",
            "details": {
                "order_id": state["order_id"],
                "purchase_date": state["purchase_date"],
                "eligible_for_refund": state["eligible_for_refund"],
                "refund_status": state["refund_status"],
                "payment_method": state["payment_method"],
                "order_amount": state["order_amount"],
            },
        }
    )

    print("human review decision", decision)

    return Command(
        update={"human_approved": decision},
        goto="refund_node" if decision else "reject_node",
    )

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
    if state["human_approved"] == False: 
        return { 
            "rejection_reason": "Human rejected refund request."
         }
    else: 
        return { 
            "rejection_reason": "Purchase period exceeded refund eligibility window."
         }

builder = StateGraph(OrderState)

builder.add_node("load_order_node", load_order_node)
builder.add_node("check_eligibility_node", check_eligibility_node)
builder.add_node("check_approval_requirement_node", check_approval_requirement_node)
builder.add_node("human_review_node", human_review_node)
builder.add_node("refund_node", refund_node)
builder.add_node("reject_node", reject_node)

builder.add_edge(START, "load_order_node")
builder.add_edge("load_order_node", "check_eligibility_node")

builder.add_conditional_edges("check_eligibility_node", conditional_edge)

builder.add_conditional_edges("check_approval_requirement_node", check_approval_requirement_router)

builder.add_edge("refund_node", END)
builder.add_edge("reject_node", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread_config = {
    "configurable": {
        "thread_id": "refund-approval",
    }
}

result = graph.invoke(
    {"order_id": "12345"},
    config=thread_config,
)

print(result)

if "__interrupt__" in result:
    interrupt_info = result["__interrupt__"][0]
    approval_request = interrupt_info.value

    print(approval_request["question"])

    for key, value in approval_request["details"].items():
        print(f"{key}: {value}")

    answer = input("Approve refund? [y/n]: ").strip().lower()
    decision = answer in {"y", "yes"}

    final_result = graph.invoke(
        Command(resume=decision),
        config=thread_config,
    )
else:
    final_result = result

print(final_result)