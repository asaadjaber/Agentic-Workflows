from langchain.agents import create_agent, AgentState
from langchain import BaseMessage

class CustomAgentState(AgentState):
    messages: List[str]
    preferences: dict

agent = create_agent(
    model="openai:gpt-5-nano",
    system_prompt="You are a helfpul assistant."
)

result = agent.invoke(
    {
        "messages": [ 
            {
                "role": "user",
                # "content": "What's the weather in San Francisco?"
                "content": "My name is Asaad. Remember this for future conversations."
            }
        ]
    }
)

print(result["messages"][-1].content_blocks)




