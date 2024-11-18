# cd tershine
# source myenv/bin/activate
# python "backend.py"
# uvicorn backend:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os
import time
import json
import requests
import pinecone
import uuid
import re
from requests.exceptions import HTTPError

# from langchain.chat_models import ChatOpenAI
# from llama_index.llms.openai import OpenAI
from langchain.agents import initialize_agent, AgentType, Tool

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool
# from langchain_community.llms import OpenAI
# from langchain_community.llms.openai import OpenAI  # Ensure langchain-community is installed
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.manager import StdOutCallbackHandler, CallbackManager

# from openai import OpenAI
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
from scipy.spatial.distance import cosine
# nltk.download('punkt_tab')

from llama_index.core.schema import TextNode, QueryBundle
# from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# tiktoken
import tiktoken

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

# # Set the tokenizer for `text-embedding-3-small`
# embedding_encoding = tiktoken.get_encoding("cl100k_base")
# embedding_model.tokenizer = embedding_encoding.encode  # Tokenizer for embedding model

# llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY,
#                  system_prompt="You are a helpful assistant. Always respond in English, regardless of the input language.")

# # set a global llm
# Settings.llm = llm

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

def split_text(text, max_length=1536):
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

vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Callback for printing thought process
callback_manager = CallbackManager([StdOutCallbackHandler()])

# Step 1: Define the Tools

# 1. RETRIEVAL tool that fetches similar documents from Pinecone
# def retrieve_from_vectorstore(query_text):
#     # Embed the query text
#     query_embedding = embedding_model.get_text_embedding(query_text)

#     # Query Pinecone for similar results
#     pinecone_results = pinecone_index.query(
#         vector=query_embedding,
#         top_k=3,
#         include_metadata=True
#     )

#     # Retrieve and combine text
#     retrieved_texts = [result['metadata']['content'] for result in pinecone_results['matches']]
#     return " ".join(retrieved_texts)

def retrieve_from_vectorstore(query_text, max_top_k=5, initial_top_k=3, similarity_threshold=0.5):
    attempt = 0
    top_k = initial_top_k
    retrieved_texts = ""
    
    # Embed the query text
    query_embedding = embedding_model.get_text_embedding(query_text)

    while top_k <= max_top_k:
        # Query Pinecone with the current top_k value
        pinecone_results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if not pinecone_results['matches']:
            print("No relevant results found in the vector store.")
            return "OUT_OF_SCOPE"  # Signal for out-of-scope query
        
        # Retrieve text and calculate similarity for the top result
        retrieved_texts = " ".join([result['metadata']['content'] for result in pinecone_results['matches']])
        top_result_vector = pinecone_results['matches'][0]['values']
        
        # Check similarity of the top result
        if top_result_vector:
            top_similarity = 1 - cosine(query_embedding, top_result_vector)
            if top_similarity < similarity_threshold:
                print(f"Similarity too low ({top_similarity:.2f}). Exiting retrieval.")
                return "OUT_OF_SCOPE"  # Signal for out-of-scope query
        
        # Check for product links in the retrieved text
        if re.search(r'https?://[^\s]+', retrieved_texts):  # Regex to check if URLs are present
            return retrieved_texts  # Signal for available product info

        # If no links found, increase top_k and try again
        print(f"Attempt {attempt + 1}: No product info found. Increasing top_k to {top_k + 1} and retrying.")
        top_k += 1
        attempt += 1

    return "OUT_OF_SCOPE"

# retrieval_tool = Tool(
#     name="pinecone_retriever",
#     func=retrieve_from_vectorstore,
#     description="Retrieves relevant documents from the Pinecone vector store."
# )
retrieval_tool = Tool(
    name="pinecone_retriever",
    func=lambda query_text: retrieve_from_vectorstore(query_text),
    description="Retrieves relevant documents from the Pinecone vector store, iteratively expanding top_k if no product links are found."
)

# # 2. RESPONSE GENERATION tool using LLM
# def generate_response(query_text, context):
#     if context == "OUT_OF_SCOPE":
#         return "The query is likely out of scope for the available context, and no relevant information was found."
        
#     query_with_context = f"{query_text}\nContext: {context}"
#     response = llm.generate(query_with_context)
#     return response


# generation_tool = Tool(
#     name="response_generator",
#     func=lambda query_text: generate_response(query_text, retrieve_from_vectorstore(query_text)),
#     description="Generates a response based on the retrieved context."
# )

# # 3. PRODUCT extraction tool
# def extract_product_info(context):
#     product_info = []
    
#     # Enhanced pattern to capture URLs with or without Markdown-style descriptions
#     url_pattern = re.compile(r'(?:\[(.*?)\]\((https?://[^\s)]+)\))|((https?://[^\s)]+))')
    
#     matches = url_pattern.findall(context)
#     for match in matches:
#         if match[0]:  # Markdown-style match with product name
#             product_name, product_link = match[0], match[1]
#             product_info.append(f"Product: {product_name}, Link: {product_link}")
#         elif match[2]:  # Direct URL without Markdown description
#             product_info.append(f"Link: {match[2]}")
    
#     return "\n".join(product_info) if product_info else "No specific product information found."

# product_extractor_tool = Tool(
#     name="product_extractor",
#     func=extract_product_info,
#     description="Extracts product information and links from the context."
# )

### FUTURE USE? CAN IGNORE FOR NOW.
# # 2. WEATHER tool for GREETING
# def get_stockholm_weather(*args, **kwargs):
#     meteo_api_key = os.getenv('METEOSOURCE_API_KEY')
#     try:
#         # Correct URL for the MeteoSource API with `place_id` usage
#         meteo_url = "https://www.meteosource.com/api/v1/free/point"
        
#         params = {
#             "place_id": "stockholm",  # Use 'stockholm' as the place_id
#             "sections": "current",
#             "language": "en",
#             "units": "metric",
#             "key": meteo_api_key
#         }

#         # Perform the API request
#         weather_response = requests.get(meteo_url, params=params)
#         weather_response.raise_for_status()
#         weather_data = weather_response.json()

#         # Extract weather details
#         weather_desc = weather_data["current"]["summary"].lower()
#         temperature = weather_data["current"]["temperature"]
#         icon_code = weather_data["current"].get("icon")  # Assuming icon code is available

#         # Mapping specific messages based on the icon code
#         if icon_code in [10, 11, 12, 13, 22]:  # Light rain, Rain, Possible rain, Rain shower, Rain and snow
#             greeting_message = f"Hey there! It’s currently {weather_desc} and {temperature}°C in Stockholm. " \
#                                f"Rain can leave water spots on your car's finish. How’s yours looking?"

#         elif icon_code in [16, 17, 18, 19, 34]:  # Light snow, Snow, Possible snow, Snow shower, Snow shower (night)
#             greeting_message = f"Hey there! It’s {weather_desc} and {temperature}°C in Stockholm. " \
#                                f"Snow can lead to salt and grime buildup—perfect time for a thorough car wash. " \
#                                f"Is your car ready for this weather?"

#         elif icon_code in [2, 3, 4, 28]:  # Sunny, Mostly sunny, Partly sunny, Partly clear (night)
#             greeting_message = f"Hey there! It's a sunny {temperature}°C in Stockholm. " \
#                                f"Great weather to make that car shine! Is it ready for a day out?"

#         elif icon_code in [5, 6, 7, 31]:  # Mostly cloudy, Cloudy, Overcast, Overcast with low clouds (night)
#             greeting_message = f"Hey there! It’s cloudy and {temperature}°C in Stockholm. " \
#                                f"A bit dull outside, but a clean car can still brighten things up. How’s yours looking?"

#         elif icon_code in [9, 15, 24, 36]:  # Fog, Local thunderstorms, Possible freezing rain, Possible freezing rain (night)
#             greeting_message = f"Hey there! It’s {weather_desc} and {temperature}°C in Stockholm. " \
#                                f"This weather can be tough on your car. Might be a good idea to give it some extra care!"

#         elif icon_code in [14, 33]:  # Thunderstorm, Local thunderstorms (night)
#             greeting_message = f"Hey there! Thunderstorms with {temperature}°C in Stockholm. " \
#                                f"This weather can be rough on cars. Consider giving it some extra care after the storm!"

#         elif icon_code in [23, 25]:  # Freezing rain, Hail
#             greeting_message = f"Hey there! It’s {weather_desc} and {temperature}°C in Stockholm. " \
#                                f"These conditions can leave stubborn residue on your car. A wash might be in order!"

#         else:
#             greeting_message = f"Hey there! It’s {weather_desc} and {temperature}°C in Stockholm. " \
#                                f"Whatever the weather, a clean car always feels great. Is yours in top shape?"

#         return greeting_message

#     except requests.RequestException:
#         return "I couldn't fetch the weather data right now. Please try again later."

# # Define the tool
# weather_tool = Tool(
#     name="stockholm_weather",
#     func=get_stockholm_weather,
#     description="Provides a greeting with Stockholm’s current weather and prompts about car cleaning."
# )

tools = [retrieval_tool]

# 1. Provide a greeting based on Stockholm’s current weather.

# Step 2: Create the Prompt

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Answer the question by retrieving relevant products from Tershine, with product links, prices, and brief justifications.
DO NOT GIVE ME IMAGES IN THE OUTPUT.

Use the following action format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: If you receive an "OUT_OF_SCOPE" message, do not attempt to answer based on general knowledge. Instead:
1. Respond that no relevant information is available in the context.
2. Suggest links or resources where the user may be able to find relevant information.

Final Answer: the final answer to the original input question crafted like a storyline with steps if necessary

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Step 3: Initialize the ReAct Agent

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)  # Assuming gpt-4o-mini model in use
agent = create_react_agent(llm, tools, prompt)

# Step 4: Set up the Agent Executor

agent_executor = AgentExecutor(agent=agent, tools=tools, callback_manager=callback_manager, handle_parsing_errors=True)

# Step 5: Run a Query through the ReAct Agent
# Request model for FastAPI
class QueryRequest(BaseModel):
    question: str

# API Endpoint for querying the ReAct Agent
@app.post("/query/")
async def query_agent(query: QueryRequest):
    try:
        # Pass the input query to the agent executor
        response = agent_executor.invoke({"input": query.question})
        
        # Extract and return the final response
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# # # Start FastAPI server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# ----- ENGLISH -----
# query_text = "How to cook an egg"
# query_text = "I have driven on salty roads how to clean my car? what products should i use"
# query_text = "I want to have gloss on my car all over"
# query_text = "what cold degreasing products should i get"
# query_text = "Recommend me a car to buy"
# query_text = "Can I use any tershine products for home cleaning?"
# query_text = "Can I use any tershine products for cleaning a wooden table?"
# query_text = "Can I use any tershine products for my horse?"
# query_text = "My bicycle is extremely dirty how do I clean it with tershine products?"
# query_text = "My kid threw up in the backseat of my car. How do i clean it and how do i get rid of the smell?"
# query_text = "My dashboard is very dirty and dusty how do I clean it? Including the steering wheel."

# ----- SWEDISH -----
# query_text = "min in´strumentbräda är dammig och smutsig hur gör jag ren den inklusive ratten"

# response = agent_executor.invoke({"input": query_text})

# # Print final output
# print("\nFinal Answer from ReAct Agent:")
# print(response['output'])


## ***** OLD DEPRECATED CODE *****
# # Initialize debug handler
# llama_debug = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([llama_debug])
# Settings.callback_manager = callback_manager

# print(vector_index)

# print("Vector index created with Pinecone as backend!")

# # Step 4: Query the Vector Index for RAG Application

# # Use the callback manager in your query engine setup
# query_engine = vector_index.as_query_engine(llm)

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

# # Print info on llm inputs/outputs - returns start/end events for each LLM call
# event_pairs = llama_debug.get_llm_inputs_outputs()
# print(event_pairs[0][0]) # Shows what was sent to LLM