from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation"

)

#01 Prompt
prompt = PromptTemplate(
    template="Write 5 intresting facts about: {topic}",
    input_variables=['topic']
)

#02 Model
model = ChatHuggingFace(llm=llm)

#03 Output Parser
parser = StrOutputParser()

# Simple Chain
chain = prompt | model | parser

result  = chain.invoke({'topic': 'Black Holes'})
print(result)

# To visualize the chain graph
chain.get_graph().print_ascii() 