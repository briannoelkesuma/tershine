# streamlit run app.py

import streamlit as st
import requests
from streamlit_chat import message  # Using streamlit-chat for chat bubbles

# Backend URL
backend_url = "http://localhost:8000/query/"

# Page configuration
st.set_page_config(
    page_title="Tershine Chatbot",
    page_icon="🧼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar logo and about section
with st.sidebar:
    st.image(
        "https://www2.bilia.se/globalassets/commerce/p-lo-tsdek30s_arb.jpg?changed=638658170600000000&width=800&height=600&quality=80&fit=bounds",
        use_column_width=True,
    )
    st.markdown("### Welcome to the Tershine Assistant 🧼")
    st.markdown(
        "Ask me anything about Tershine products, car cleaning, or maintenance. I’ll provide personalized and expert advice!"
    )
    st.markdown("---")
    st.markdown("#### Quick Tips:")
    st.markdown("- For product recommendations, ask 'Which degreaser should I use?'\n"
                "- To learn about car cleaning steps, try 'How do I clean my dashboard?'\n"
                "- Available in **English** and **Swedish**!")

# Main header
st.title("Tershine Chatbot Interface 🧼")
st.markdown("### Chat with Tershine Assistant")

# Chat container
chat_container = st.container()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to display the chat history
def display_chat_history():
    with chat_container:
        for i, entry in enumerate(st.session_state["chat_history"]):
            user_query, bot_response = entry

            # Display user query as a right-aligned chat bubble
            message(user_query, is_user=True, key=f"user_query_{i}")

            # Display bot response as a left-aligned chat bubble
            message(bot_response, is_user=False, key=f"bot_response_{i}")

# Display the existing chat history
display_chat_history()

# Chat input container at the bottom
with st.container():
    # Create columns for input field and send button
    input_col, button_col = st.columns([8, 1])

    with input_col:
        # Text input for the user message
        query_text = st.text_input(
            "",
            placeholder="Type your question here...",
            key=f"input_{len(st.session_state['chat_history'])}",
            label_visibility="collapsed",
        )

    with button_col:
        # Send button with icon
        send_button = st.button("💬 Send", help="Click to send your message")

# Send query to the backend when Enter is pressed or Send button is clicked
if query_text.strip() and (st.session_state.get(f"send_trigger_{len(st.session_state['chat_history'])}", False) or send_button):
    # Add thinking placeholder to the chat history
    st.session_state["chat_history"].append((query_text, "🤔 Tershine Assistant is thinking..."))

    # Send the query to the backend
    response = requests.post(backend_url, json={"question": query_text})

    if response.status_code == 200:
        # Get the response text
        bot_response = response.json().get("response", {}).get("output", "Sorry, I couldn't find an answer.")
        # Update the thinking placeholder with the real response
        st.session_state["chat_history"][-1] = (query_text, bot_response)
    else:
        st.session_state["chat_history"][-1] = (query_text, "Sorry, there was an error processing your request.")

    # Refresh the chat
    st.rerun()

# Enhanced CSS for polished UI
st.markdown(
    """
    <style>
    .stTextInput {
        font-size: 16px;
        margin-bottom: 0px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 16px;
        cursor: pointer;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stChatContainer {
        max-height: 60vh;
        overflow-y: auto;
        padding: 10px;
    }
    .stMarkdown {
        font-family: 'Montserrat', sans-serif;
        line-height: 1.5;
    }
    .stSidebar > div {
        padding: 15px;
    }
    .stContainer {
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)