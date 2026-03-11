from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_core.parsers import JsonOutputParser

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text.\n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({"topic": "black hole"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

print("Detailed Report: ", result1.content)
print("Summary: ", result2.content)