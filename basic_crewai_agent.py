import os
from crewai import Agent, Task, Crew, LLM
import creds

os.environ["ANTHROPIC_API_KEY"] = creds.anthropic_api_key

llm = LLM(model="claude-haiku-4-5")

info_agent = Agent(
    role = 'Information Agent',
    goal = 'Give compelling information about the topic mentioned in detail.',
    backstory = '''
You love to research and know information. people love and hate you for your knowledge. You are a great researcher and can find information on any topic. You are very good at finding information and giving it in a compelling way. You are also very good at finding information that is not easily found.''',
    llm = llm
)

task1 = Task(
    description='Tell me interesting facts about the IPL 2020 season.',
    expected_output='Give me a quick summary and then give me 5 bullet points with interesting facts about the IPL 2020 season.',
    agent=info_agent
)

crew = Crew(
    agents=[info_agent],
    tasks=[task1],
    verbose=True
)

result = crew.kickoff()
print('####### RESULT #######\n', result)
