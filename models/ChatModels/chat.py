from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

from langchain_groq import ChatGroq
model = ChatGroq(api_key=GROQ_API_KEY, 
                 model='llama-3.1-8b-instant',
                 temperature=0.7) # vary from 0.1 to 1.5 create randomness in the output.
result = model.invoke("What is the best way to get rid of a bad habit? Make it simple and short just points and one line description for each point.")
print(result.content)