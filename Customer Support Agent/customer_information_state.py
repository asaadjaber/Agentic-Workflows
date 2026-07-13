from langchain.agents import AgentState

class CustomerInformationState(AgentState): 
    first_name: str
    customer_id: str
    order_id: str
    current_issue: str
    tracking_number: str