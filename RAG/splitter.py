import warnings
warnings.filterwarnings('ignore')
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('DS.pdf')
docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 1,
    separator = ""
)
result = splitter.split_documents(docs)
print(result[0]) # show only 1st chunk

text = """
Artificial Intelligence (AI) is one of the most transformative technologies of the 21st century, fundamentally reshaping industries, economies, and societies across the globe. At its core, AI refers to the simulation of human intelligence processes by computer systems, enabling machines to perform tasks that traditionally required human cognition such as learning, reasoning, problem-solving, perception, and language understanding. The history of AI dates back to the 1950s when Alan Turing, a British mathematician and computer scientist, proposed the famous Turing Test as a measure of machine intelligence, asking whether a machine could exhibit intelligent behavior indistinguishable from that of a human. Since then, AI has evolved through multiple phases — from early rule-based expert systems in the 1970s and 1980s, to the machine learning revolution of the 1990s and 2000s, and finally to the deep learning era that began in the 2010s and continues to dominate today. Deep learning, a subset of machine learning inspired by the structure of the human brain, uses artificial neural networks with many layers to automatically learn patterns from vast amounts of data, achieving superhuman performance in areas like image recognition, speech processing, and natural language understanding. The rise of large language models such as GPT, LLaMA, Claude, and Gemini has further pushed the boundaries of what AI can do, enabling systems to write essays, generate code, analyze legal documents, compose music, create artwork, and hold meaningful conversations with humans. In Pakistan and other developing nations, AI presents both enormous opportunity and significant challenge — on one hand, it can accelerate economic growth, improve healthcare delivery, enhance agricultural productivity, and democratize access to quality education; on the other hand, it raises serious concerns about job displacement, digital inequality, data privacy, algorithmic bias, and the concentration of technological power in the hands of a few wealthy nations and corporations. As the world races toward an AI-powered future, it becomes increasingly important for students, developers, policymakers, and citizens to develop a deep understanding of how these systems work, what their limitations are, and how we can harness their power responsibly and equitably for the benefit of all humanity.
"""

c_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0
)
chunks = c_splitter.split_text(text)
print(chunks)
print(len(chunks))
