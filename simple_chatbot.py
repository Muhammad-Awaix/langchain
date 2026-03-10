from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv 
import os
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model = ChatGroq(api_key=GROQ_API_KEY, 
                 model='llama-3.1-8b-instant',
                 temperature=0.5)

# giving the context of the conversation to the model
chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. Goodbye!")
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ",response.content)

print(chat_history)