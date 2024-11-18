# cd tershine
# source myenv/bin/activate
# python "backend.py"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os
import time
import json
import requests
import pinecone
import uuid
from requests.exceptions import HTTPError

from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, get_response_synthesizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from pinecone import ServerlessSpec
from nltk import sent_tokenize
import nltk
# nltk.download('punkt_tab')

from llama_index.core.schema import TextNode, QueryBundle
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Load API keys
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Firecrawl and headers setup
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# Set up Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbedding(api_key=OPENAI_API_KEY, model='text-embedding-3-small')

llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.1,
                 system_prompt="You are tershine's (car wash company) helpful enthusiast assistant. "
                 "Always respond in English, regardless of the input language. "
                 "DO NOT recommend specific products, provide prices, or suggest links unless EXPLICITLY MENTIONED in CONTEXT given!"
                 "Communicate in a way that's friendly with a fun tone, using a conversational style as seen in Tershine's social media."
                 "Recommend RELEVANT tershine products and even those in combination whenever applicable."
                 "Output Format for this should be like Product: Title with Product Links from the context embedded \n and provide the reason why without any formatting."
                 "Ensure to always provide product links if a product is recommended!"
                 "If the response contains product information, make sure to include the links and reasons below so that users can be directed to them based on the guides!"
                 "Separate and structure the content neatly in a readable format with proper formatting"
                 "Product links should be solely based on the CONTEXT given and DO NOT REPHRASE OR CHANGE ANY LINKS!"
                 "Flygrostbottagare is a typo in the context, it should be Flygrostborttagare"
                 )

# set a global llm
Settings.llm = llm

# List to store content for vector database
raw_documents = []

# # TODO 1: Uncomment to scrape all links and Replace with the washing_guide_links.txt
# # Step 1: Scrape each link for detailed content
# with open("washing_guide_links.txt", "r") as file:
#     links = file.readlines()

# for link in links:
#     link = link.strip()
#     success = False

#     while not success:
#         try:
#             scrape_result = firecrawl_app.scrape_url(link, params={'formats': ['markdown']})

#             # Append the scrape result to raw_documents if successful
#             raw_documents.append({
#                 "title": scrape_result['metadata']['title'], 
#                 "description": scrape_result['metadata']['description'], 
#                 "language": scrape_result['metadata']['language'],
#                 "markdown": scrape_result["markdown"]
#             })
#             success = True  # Exit loop after successful request

#         except HTTPError as e:
#             # Check if the error is due to rate limiting (429)
#             if e.response.status_code == 429:
#                 print("Rate limit exceeded. Waiting for 15 seconds before retrying...")
#                 time.sleep(15)  # Wait 15 seconds before retrying
#             else:
#                 print(f"Failed to scrape {link}: {e}")
#                 break  # Exit loop on other HTTP errors

# print("Scraping completed!")

# # Step 2: Save scraped content to JSON
# with open("scraped_data_all.json", "w") as file:
#     json.dump(raw_documents, file, indent=4)

# # temporary to test the rag pipeline: Load the data from the JSON file
# with open("scraped_data.json", "r") as file:
#     raw_documents = json.load(file)

with open("scraped_data_all.json", "r") as file:
    raw_documents = json.load(file)

# # Connect to Pinecone index
index_name = "tershine"

# # Check if the index exists, and create if it doesn’t
# if index_name not in pc.list_indexes():
#     print(f"Creating Pinecone index '{index_name}' as it does not exist.")
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
# else:
#     print(f"Pinecone index '{index_name}' already exists.")

# Connect to the Pinecone index
pinecone_index = pc.Index(index_name)
print(f"Connected to Pinecone index '{index_name}'.")
print(pc.list_indexes())

# def split_text(text, max_length=1536):
#     sentences = sent_tokenize(text)
#     chunks, current_chunk = [], ""
#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) > max_length:
#             chunks.append(current_chunk)
#             current_chunk = ""
#         current_chunk += sentence + " "
#     if current_chunk:
#         chunks.append(current_chunk)
#     return chunks

# TODO: Uncomment when need to upload new embeddings!
# Step 3: Use LlamaIndex with OpenAI Embeddings for Vector Database

# # Create and store embeddings in Pinecone
# for doc in raw_documents:
#     # Split text into chunks for better embedding handling
#     text_chunks = split_text(doc['markdown'])
    
#     # Loop through chunks to generate and upsert embeddings
#     for i, chunk in enumerate(text_chunks):
#         # Generate embedding for each chunk
#         embedding = embedding_model.get_text_embedding(chunk)
        
#         # Create a unique ID for each chunk (using title + chunk index)
#         doc_id = f"{uuid.uuid4()}_{i}"  # Ensures a unique, ASCII-compliant ID
        
#         # Upsert embedding into Pinecone
#         pinecone_index.upsert([(doc_id, embedding, {
#             "content": chunk,
#             "title": doc.get("title"),
#             "description": doc.get("description"),
#             "language": doc.get("language")
#         })])

# print("All embeddings stored in Pinecone!")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="content")

# Create the VectorStoreIndex using Pinecone as the backend
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Initialize debug handler
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

print(vector_index)

print("Vector index created with Pinecone as backend!")

# Step 4: Query the Vector Index for RAG Application
# Use the callback manager in your query engine setup
query_engine = vector_index.as_query_engine(llm)

# event_pairs = llama_debug.get_llm_inputs_outputs()
# print(event_pairs[0][0]) # Shows what was sent to LLM

# Request model for FastAPI
class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def query(request: QueryRequest):
    # query_text = request.question + "Using this context, recommend RELEVANT tershine products and even those in combination whenever applicable with PRODUCT LINKS embedded in the product title and PRICES listed below and justify reasons!"
    query_text = request.question
    query_embedding = embedding_model.get_text_embedding(query_text)
    
    # Query Pinecone with the embedding
    pinecone_results = pinecone_index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Prepare context from Pinecone results
    retrieved_texts = [result['metadata']['content'] for result in pinecone_results['matches']]
    query_context = " ".join(retrieved_texts)
    print(retrieved_texts)

    # Get response from LLM
    response = query_engine.query(query_text + query_context)
    return {"response": response}

# Start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# # Define the query and embed it
# query_text = "What is the difference between alkaline degreasing and cold degreasing? Give a detailed breakdown"
# # query_text = "Vad är skillnaden mellan alkalisk avfettning och kall avfettning?"
# query_embedding = embedding_model.get_text_embedding(query_text)

# pinecone_results = pinecone_index.query(
#     vector=query_embedding,
#     top_k=3,  # Number of similar results to retrieve
#     include_metadata=True  # Retrieve metadata to access document text
# )

# for i, match in enumerate(pinecone_results['matches'], 1):
#     print(f"\n--- Retrieved Context {i} ---")
#     print(match)

# retrieved_texts = [result['metadata']['content'] for result in pinecone_results['matches']]

# query_context = " ".join(retrieved_texts)

# response = query_engine.query(query_context)

# print(response)