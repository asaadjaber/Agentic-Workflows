from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from tools.check_shipping_status_tool import check_shipping_status
from tools.fetch_customer_information_tool import fetch_customer_information
from tools.fetch_order_information_tool import fetch_order_information
from tools.update_current_issue_tool import update_current_issue
from tools.update_order_id_tool import update_order_id
from tools.refund_order import refund_order

from customer_information_state import CustomerInformationState
from middleware import add_refund_policy_context

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
           update_order_id, 
           fetch_order_information, 
           update_current_issue, 
           check_shipping_status,
           refund_order],
    middleware=[add_refund_policy_context],
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