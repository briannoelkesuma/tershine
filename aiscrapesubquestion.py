import os
import time
import json
import requests
import pinecone

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from pinecone import ServerlessSpec
from nltk import sent_tokenize
import nltk
# nltk.download('punkt_tab')

import nest_asyncio
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

# Apply nest_asyncio to allow asynchronous execution in Jupyter or similar environments
nest_asyncio.apply()

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

load_dotenv()

# Load API keys
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Firecrawl and headers setup
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# Set up Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# # Connect to Pinecone index
index_name = "tershine"  # The name of your index shown in the screenshot
# if index_name not in pc.list_indexes():
#     pc.create_index(index_name, 
#                     dimension=3072, 
#                     metric="cosine", 
#                     spec=ServerlessSpec(
#     cloud="aws",
#     region="us-east-1"
#   )
#   )  # Ensure dimensions match

# List to store content for vector database
raw_documents = []

# TODO 1: Uncomment to scrape all links and Replace with the washing_guide_links.txt
# # Step 1: Scrape each link for detailed content
# with open("test_washing_guide_links.txt", "r") as file:
#     links = file.readlines()

# for link in links:
#     link = link.strip()

#     scrape_result = app.scrape_url(link, params={'formats': ['markdown']})

#     documents.append({
#         "title": scrape_result['metadata']['title'], 
#         "description": scrape_result['metadata']['description'], 
#         "language": scrape_result['metadata']['language'],
#         "markdown": scrape_result["markdown"]
#     })

# # Step 2: Save scraped content to JSON
# with open("scraped_data.json", "w") as file:
#     json.dump(documents, file, indent=4)

# temporary to test the rag pipeline: Load the data from the JSON file
with open("scraped_data.json", "r") as file:
    raw_documents = json.load(file)

# Set up Pinecone vector store
pinecone_index = pc.Index(index_name)
# Check Pinecone index exists and is healthy
if not pinecone_index:
    raise Exception(f"Pinecone index {index_name} does not exist.")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

def split_text(text, max_length=3072):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Create Document objects with split content and minimal metadata
documents = []
for doc in raw_documents:
    text_chunks = split_text(doc['markdown'])
    for chunk in text_chunks:
        documents.append(Document(text=chunk, 
                                  metadata={
                                    "title": doc.get("title"), 
                                    "description": doc.get("description"), 
                                    "language": doc.get("language")
                                    }))

# Step 3: Use LlamaIndex with OpenAI Embeddings for Vector Database
# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(api_key=OPENAI_API_KEY, model='text-embedding-3-large')

# Create the VectorStoreIndex using Pinecone as the backend
index = VectorStoreIndex.from_documents(documents, embedding_model=embedding_model, vector_store=vector_store, chunk_size=3072)
print(index)

print("Vector index created with Pinecone as backend!")

# Step 4: Query the Vector Index for RAG Application

# Example query
# query_engine = index.as_query_engine()
# query = "What is cold degreasing?" + ". Recommend products with links to them."
# # query = "What is cold degreasing?" + "Give me products that i should get including those products that are used in combination according to my question with links to them and justify why these products so that users make an informed decision weighing the pros and cons!"
# response = query_engine.query(query)

# print("Query Response:", response)


# # SUBQUERY engine
# Set up a base query engine tool for the vector index
vector_query_engine = index.as_query_engine()
query_engine_tool = QueryEngineTool(
    query_engine=vector_query_engine,
    metadata=ToolMetadata(
        name="tershine_docs",
        description="Tershine documents and knowledge base"
    ),
)

#  Initialize the SubQuestionQueryEngine with the query engine tool
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[query_engine_tool],
    use_async=True,
)

# Example query
query = "What is cold degreasing, recommend ALL related and tershine products used in combination with clickable links for purchase in point form. Justify why as well."
# query = "What is cold degreasing, recommend related tershine products with clickable links for purchase in point form"
# query = "Vad är kallavfettning? Rekommendera tershine-produkter och inkludera länkar för köp"
response = query_engine.query(query)

print("Query Response:", response)