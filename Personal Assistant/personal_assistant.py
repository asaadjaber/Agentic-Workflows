from langchain.agents import create_agent, AgentState
from langchain_core.messages import BaseMessage

from collections.abc import Callable
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage

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

agent = create_agent(
    model="openai:gpt-5-nano",
    system_prompt="You are a helfpul assistant.",
    state_schema=CustomAgentState,
    middleware=[add_context]
)

result = agent.invoke(
    {
        "messages": [ 
            {
                "role": "user",
                "content": "Hi. Can you tell me what my name and preferred language are?"
            }
        ],
        "preferred_language": "English (U.K.)",
        "verbosity": "concise",
        "tone": "friendly",
        "name": "Asaad"
    }
)

print(result["messages"][-1].content_blocks)





