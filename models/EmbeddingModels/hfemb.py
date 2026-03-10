from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
text = "What are the prerequisites for learning langchain? Simple points only."
embeddings_result = embeddings.embed_query(text)
print(embeddings_result)
# This will create a list of floats representing the embedding of the input text.