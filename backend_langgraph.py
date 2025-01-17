# uvicorn backend_langgraph:app --reload --host 0.0.0.0 --port 8000

from langchain import hub
from langchain.agents import Tool, create_react_agent
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

from pydantic import BaseModel
from langdetect import detect
from fastapi import FastAPI
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
import re


load_dotenv()

app = FastAPI()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "tershine"  # The name of your index shown in the screenshot
pinecone_index = pc.Index(index_name)

embedding_model = OpenAIEmbedding(api_key=OPENAI_API_KEY, model='text-embedding-3-small')

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

        print(pinecone_results['matches'])
        
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

retrieval_tool = Tool(
    name="pinecone_retriever",
    func=lambda query_text: retrieve_from_vectorstore(query_text),
    description="Retrieves relevant documents from the Pinecone vector store when there are product links found."
)

tools = [retrieval_tool]

template = '''Answer the following questions as best you can in the language of the input ONLY. 

It is either swedish or english only.

Example:
1. Input: Vilken typ av avfettning ska jag använda? in sv then you will reply in SWEDISH!
2. Input: What type of degreaser should I use? in en then you will reply in ENGLISH!

You have access to the following tools:

{tools}

Chat history:
{chat_history}

Use the following format:

Answer the question as best as you can, but always provide the product title as embedded links from the tools, and brief justifications for why the product is recommended.

DO NOT GIVE ME IMAGES IN THE OUTPUT. 

Use the following action format, ensure each step is DONE before going to the next step:

Question: the input question you must answer in the language of the input ONLY.
Thought: you should always think about what to do, DO NOT REPEAT THE SAME THOUGHT.
Action: the action to take, should be one of [{tool_names}] which is retrieval_tool ONLY.
Action Input: the input to the action; DO NOT REPEAT THE SAME ACTION INPUT.
Observation: the result of the action. Check if observation matches the goal of Tershine products. Do not infer or create links that are not explicitly mentioned in the retrieved data. Ensure that product links are directly retrieved from the context provided by the Pinecone retriever tool.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: If you receive an "OUT_OF_SCOPE" message, do not attempt to answer based on general knowledge. Instead:
1. Respond that no relevant information is available in the context.
2. Suggest links or resources where the user may be able to find relevant information.
Thought: You must always conclude with a clear and concise Final Answer, even if no action could be taken.
Final Answer: the final answer to the original input question crafted like a storyline with step by step instructions and guide to help answer the question and structure your answer in point form and paragraph.

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4-turbo-preview", api_key=OPENAI_API_KEY)  # Assuming gpt-4o-mini model in use

agent_runnable = create_react_agent(llm, tools, prompt)


from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated, Sequence


class AgentState(TypedDict):
   input: str
   chat_history: List[BaseMessage]
   agent_outcome: Union[AgentAction, AgentFinish, None]
   return_direct: bool
   intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]


from langgraph.prebuilt.tool_executor import ToolExecutor


tool_executor = ToolExecutor(tools)


from langchain_core.agents import AgentActionMessageLog



def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}



from langgraph.prebuilt import ToolInvocation

def execute_tools(state):

    messages = [state['agent_outcome'] ]
    last_message = messages[-1]
    ######### human in the loop ###########   
    # human input y/n 
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    # state_action = state['agent_outcome']
    # human_key = input(f"[y/n] continue with: {state_action}?")
    # if human_key == "n":
    #     raise ValueError
    
    tool_name = last_message.tool
    arguments = last_message
    if tool_name == "Search":
        
        if "return_direct" in arguments:
            del arguments["return_direct"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input= last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(state['agent_outcome'],response)]}

    
def should_continue(state):

    messages = [state['agent_outcome'] ] 
    last_message = messages[-1]
    if "Action" not in last_message.log:
        return "end"
    else:
        arguments = state["return_direct"]
        if arguments is True:
            return "final"
        else:
            return "continue"
        

def first_agent(inputs):
    action = AgentActionMessageLog(
      tool="Search",
      tool_input=inputs["input"],
      log="",
      message_log=[]
    )
    return {"agent_outcome": action}


from langgraph.graph import END, StateGraph


workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.add_node("final", execute_tools)
# uncomment if you want to always calls a certain tool first
# workflow.add_node("first_agent", first_agent)


workflow.set_entry_point("agent")
# uncomment if you want to always calls a certain tool first
# workflow.set_entry_point("first_agent")

workflow.add_conditional_edges(

    "agent",
    should_continue,

    {
        "continue": "action",
        "final": "final",
        "end": END
    }
)


workflow.add_edge('action', 'agent')
workflow.add_edge('final', END)
# uncomment if you want to always calls a certain tool first
# workflow.add_edge('first_agent', 'action')
workflow_app = workflow.compile()

# inputs = {"input": "what is the weather in Taipei", "chat_history": [],"return_direct": False}

# for s in app.stream(inputs):
#     print(list(s.values())[0])
#     print("----")

# Define the input model for the endpoint
class QueryRequest(BaseModel):
    question: str

# Language detection helper
def identify_language(query: str) -> str:
    """Identify the language of the query using langdetect."""
    try:
        language = detect(query)
        return language
    except Exception as e:
        print(f"Error identifying language: {e}")
        return "en"  # Default to English if detection fails

# FastAPI endpoint
@app.post("/query/")
async def query_agent(query: QueryRequest):
    try:

        # Prepare inputs for the workflow
        inputs = {
            "input": f"{query.question.strip()}",
            "chat_history": [],  # Initialize chat history
            "return_direct": False,  # Allow intermediate steps
        }

        final_output = None

        # Run the workflow and collect the response
        response_content = ""
        for state in workflow_app.stream(inputs):
            # Stream responses and print intermediate outputs
            response_content = list(state.values())[0]  # Final response
            print("Response: ", response_content)

            if 'agent_outcome' in response_content:
                if isinstance(response_content["agent_outcome"], AgentFinish):
                    final_output = response_content["agent_outcome"].return_values["output"]
                    print("Final Output Found: ", final_output)

        return {"response": final_output}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "response": "An error occurred while processing your request. Please try again later."
        }