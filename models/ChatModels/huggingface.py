from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # ✅ supported model
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")  # ✅ token
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What are the prerequisites for learning LangChain? Simple points only.")
print(result.content)