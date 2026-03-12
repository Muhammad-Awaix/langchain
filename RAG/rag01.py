import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader

#01 PDF loaders
loader = DirectoryLoader(
    path = "pdf's",
    glob = "*.pdf",
    loader_cls=PyPDFLoader
)

# also use the load()
docs = loader.lazy_load()
for documents in docs:
    print(documents.metadata)

#02 Web base loader 
url = "https://docs.langchain.com/oss/python/langchain/overview?search=document"

web_loader = WebBaseLoader(url)

load1 = web_loader.load()
print(load1[0].page_content)