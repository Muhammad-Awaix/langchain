from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me 5 facts about {topic} \n{format_inst}",
    input_variables=['topic'],
    partial_variables={'format_inst': parser.get_format_instructions()}
)
chain = template |model |parser
result = chain.invoke({'topic': 'Black Holes'})
print(result)

# Also there are many other output parsers available in langchain_core.output_parsers, such as:
# - StandardOutputParser
# - JsonKeyOutputParser
# - CommaSeparatedListOutputParser