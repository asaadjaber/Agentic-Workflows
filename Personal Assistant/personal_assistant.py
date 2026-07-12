from langchain.agents import create_agent, AgentState
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from collections.abc import Callable
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage

from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

class CustomAgentState(AgentState):
    preferred_language: str
    verbosity: str
    tone: str
    name: str

@wrap_model_call
def add_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    new_content = list(request.system_message.content_blocks) + [ 
        {
            "type": "text", "text": f"""User name is {request.state["name"]}. 
                                        Preferred language is {request.state["preferred_language"]}.
                                        Verbosity is {request.state["verbosity"]}.
                                        Tone is {request.state["tone"]}.
                                    """,
        }
    ]
    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))

@tool
def set_preferred_language(new_language: str, runtime: ToolRuntime[None, CustomAgentState]) -> Command:
    """Set the user's preferred language in the conversation state."""
    return Command(
        update={
            "preferred_language": new_language,
            "messages": [
                ToolMessage(
                    content=f"Preferred language set to {new_language}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

agent = create_agent(
    model="openai:gpt-5-nano",
    system_prompt="You are a helfpul assistant.",
    state_schema=CustomAgentState,
    middleware=[add_context],
    checkpointer=InMemorySaver(),
    tools=[set_preferred_language]
)

thread_config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {
        "messages": [ 
            {
                "role": "user",
                "content": "Hi. What are my name and preferred language?"
            }
        ],
        "preferred_language": "English",
        "verbosity": "concise",
        "tone": "friendly",
        "name": "Asaad"
    }, 
    thread_config
)

print(result["messages"][-1].content_blocks)

result = agent.invoke(
    {
        "messages": [ 
            {
                "role": "user",
                "content": "Set my preferred language to English (U.K.)"
            }
        ],
    }, 
    thread_config
)

print(result["messages"][-1].content_blocks)

result = agent.invoke(
    {
        "messages": [ 
            {
                "role": "user",
                "content": "What's my preferred language?"
            }
        ],
    }, 
    thread_config
)

print(result["messages"][-1].content_blocks)

