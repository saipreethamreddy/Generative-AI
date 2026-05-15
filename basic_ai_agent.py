import os
import creds
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import Tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", creds.anthropic_api_key)

llm = ChatAnthropic(model="claude-haiku-4-5")

search_tool = DuckDuckGoSearchRun()

react_template = """
Answer the following questions as best you can. You have access to the following tools:
 
{tools}
 
Use the following format:
 
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
 
Begin!
 
Question: {input}
Thought:{agent_scratchpad}
"""
 
#prompt = PromptTemplate.from_template(react_template)

agent = create_agent (
    model = llm,
    tools = [search_tool],
    system_prompt=react_template
)

response = agent.invoke({
    "messages": [HumanMessage(content="3 ways to reach Hyderabad from Bangalore")]
})

# 4. Print the final result
print(response["messages"][-1].content)
