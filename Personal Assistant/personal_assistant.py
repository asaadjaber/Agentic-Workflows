from langchain.agents import create_agent, AgentState
from langchain_core.messages import BaseMessage

class CustomAgentState(AgentState):
    preferred_language: str
    verbosity: str
    tone: str
    name: str

agent = create_agent(
    model="openai:gpt-5-nano",
    system_prompt="You are a helfpul assistant.",
    state_schema=CustomAgentState
)

result = agent.invoke(
    {
        "messages": [ 
            {
                "role": "user",
                "content": "Hi. Can you tell me what my name and preferred_language are?"
            }
        ],
        "preferred_language": "English (U.K.)",
        "verbosity": "concise",
        "tone": "friendly",
        "name": "Asaad"
    }
)

print(result["messages"][-1].content_blocks)




