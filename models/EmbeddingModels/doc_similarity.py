from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
documents = ["Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed.",
"Deep learning uses neural networks with many layers to analyze large amounts of data, powering applications like image recognition and natural language processing.",
"Python is the most popular programming language for data science due to its simplicity, rich libraries like NumPy and Pandas, and strong community support.",
"Climate change is causing rising sea levels, extreme weather events, and disruptions to ecosystems, making it one of the most urgent global challenges today.",
"Football is the world's most popular sport, played in over 200 countries, with the FIFA World Cup being the most watched sporting event on the planet."]

query  = "Tell me about the Climate change"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(f"Most similar document: {documents[index]}")
