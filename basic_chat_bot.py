from langchain_anthropic import ChatAnthropic
import creds
import os

os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", creds.anthropic_api_key)

model = ChatAnthropic(model="claude-haiku-4-5")

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break

    response = model.invoke(chat_history)
    chat_history.append(response.content)
    print("Chatbot:", response.content)
