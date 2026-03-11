from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
# from langchain.core.output_parsers import PydanticOutputParser

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Clearify the sentiment of the follwing feedback text into positive or negative \n{feedback}",
    input_variables=['feedback']
)
classifier_chain = prompt1 | model | parser

# result = classifier_chain.invoke({'feedback':"This product is very good"})
# print(result)
# for strict the output you can use the pydanticoutput parser 
# create a pydantic class then add in the partial_variables

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive response \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative response \n {feedback}",
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x: 'positive' in x['sentiment'].lower(), prompt2 | model | parser),
    (lambda x: 'negative' in x['sentiment'].lower(), prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | RunnableLambda(lambda x: {'sentiment': x}) | branch_chain

result = chain.invoke({'feedback': 'This is an amazing course'})
print(result)




