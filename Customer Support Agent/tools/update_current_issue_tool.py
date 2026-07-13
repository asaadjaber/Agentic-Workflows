from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from customer_information_state import CustomerInformationState
from langgraph.types import Command

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