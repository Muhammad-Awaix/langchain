from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation"
)


#01 2 Models
model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

#02 Prompts

prompt1 = PromptTemplate(
    template = "Generate a short and simple notes from the follwing text \n {text}",
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template="Generate a 5 quiz questions from the following text \n{text}",
    input_variable = ['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document: \n{notes} and {quiz}",
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
    A black hole is one of the most mysterious and extreme objects in the universe, formed when a very massive star collapses under its own gravity at the end of its life. When the core of such a star runs out of nuclear fuel, it can no longer support itself against gravity, causing it to collapse inward until an incredibly dense point called a singularity is formed. Around this singularity lies a boundary known as the event horizon, which is the point beyond which nothing can escape—not even light. Because light cannot escape, black holes appear completely dark, making them invisible directly; scientists detect them by observing how they affect nearby stars, gas, and light. The gravitational pull of a black hole is so strong that matter falling toward it forms a rapidly spinning disk called an accretion disk, which heats up and emits intense radiation before crossing the event horizon. Some black holes also produce powerful jets of high-energy particles that shoot out from their poles at nearly the speed of light. Black holes come in several sizes: stellar black holes formed from collapsing stars, intermediate black holes that are larger but still mysterious, and supermassive black holes that sit at the centers of most galaxies, including the one in the center of our own Milky Way galaxy. These supermassive black holes can contain millions or even billions of times the mass of our Sun. Despite their name, black holes do not “suck” everything around them like cosmic vacuum cleaners; objects must come very close before being captured by their gravity. Modern physics continues to study black holes to understand gravity, spacetime, and the fundamental laws of the universe, with theories such as Hawking radiation suggesting that black holes may slowly lose energy and eventually evaporate over extremely long periods of time. Because they combine the effects of extreme gravity, quantum mechanics, and cosmic evolution, black holes remain one of the most fascinating and actively researched phenomena in astrophysics.
"""

result = chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()
