# import streamlit as st
# import requests  # Import the requests library

# # Custom icons for user and assistant
# user_icon = "https://cdn-icons-png.flaticon.com/512/6897/6897018.png"
# assistant_icon = "https://cdn.handshake.fi/images/autodudese/produktbilder/tershine/logo-tershine.png"

# # Suggestions for users
# SUGGESTIONS = [
#     "Vilken typ av avfettning ska jag använda?",
#     "Hur polerar jag min bil?",
#     "Vad är det bästa sättet att vaxa min bil?",
#     "Hur rengör jag min bil efter snö?"
# ]

# # Backend URLs
# BACKEND_URL = "http://localhost:8000/query/"  # Replace with your backend URL


# def main():
#     initialise_chat()
#     display_suggestions()
#     display_chat_history()
#     handle_user_input()


# def initialise_chat():
#     """Initialize the chat history and suggestions visibility."""
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     if "suggestions_visible" not in st.session_state:
#         st.session_state.suggestions_visible = True

# def display_suggestions():
#     """Display suggested prompts above the chat input."""
#     if st.session_state.suggestions_visible and not st.session_state.chat_history:
#         st.subheader("Need some ideas? Try these:")
#         cols = st.columns(len(SUGGESTIONS))
#         for i, suggestion in enumerate(SUGGESTIONS):
#             if cols[i].button(suggestion, key=f"suggestion_{i}"):
#                 # Treat clicked suggestion as user input
#                 add_user_input(suggestion)
#                 st.session_state.suggestions_visible = False  # Hide suggestions
#                 st.rerun()


# def display_chat_history():
#     """Render chat history with appropriate roles and icons."""
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             with st.chat_message("user", avatar=user_icon):
#                 st.markdown(message["content"])
#         elif message["role"] == "assistant":
#             with st.chat_message("assistant", avatar=assistant_icon):
#                 st.markdown(message["content"])


# def handle_user_input():
#     """Handle manual user input from the chat box."""
#     user_input = st.chat_input("Ask me anything about car care :)")
#     if user_input:
#         add_user_input(user_input)


# def add_user_input(user_input):
#     """Add user input to chat history and generate a response."""
#     # Append user input to the chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user", avatar=user_icon):
#         st.markdown(user_input)

#     # Generate and display assistant response
#     generate_assistant_response(user_input)


# def generate_assistant_response(user_input):
#     """Generate assistant response by querying the backend."""
#     with st.chat_message("assistant", avatar=assistant_icon):
#         # Call the backend API
#         response = requests.post(BACKEND_URL, json={"question": user_input})

#         if response.status_code == 200:
#             response_data = response.json()

#             # Handle response output
#             assistant_response = response_data.get("response", "I'm sorry, I couldn't retrieve the information.")
#             if isinstance(assistant_response, dict):
#                 assistant_response = assistant_response.get("output", "Sorry, I couldn't process that.")

#             # Add assistant response to chat history
#             st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
#             st.markdown(assistant_response)
#         else:
#             st.markdown("Error: Unable to get response from the server.")


# if __name__ == "__main__":
#     st.set_page_config(page_title="Tershine Chatbot", page_icon="🧼")
#     st.title("Chat with Tershine Assistant 🧼")
#     st.subheader("Your expert car care assistant")
#     st.markdown(
#         """
#         Welcome to the Tershine Assistant! 🚗  
#         Ask anything about car cleaning, maintenance, or Tershine products.
#         """
#     )
#     main()

import os
import streamlit as st
import requests

user_icon = "https://cdn-icons-png.flaticon.com/512/6897/6897018.png"
assistant_icon = "https://cdn.handshake.fi/images/autodudese/produktbilder/tershine/logo-tershine.png"

# Suggestions for users
GENERAL_QUESTIONS = [
    "Vilken typ av avfettning ska jag använda?",
    "Hur polerar jag min bil?",
    "Vad är det bästa sättet att vaxa min bil?",
    "Hur rengör jag min bil efter snö?"
]

BACKEND_URL = "http://localhost:8000/query/"  # Replace with your backend URL

def main():
    initialise_chat()
    display_suggestions()
    display_chat_history()
    handle_user_input()


def initialise_chat():
    """Initialize the chat history and suggestions visibility."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "suggestions_visible" not in st.session_state:
        st.session_state.suggestions_visible = True


def display_suggestions():
    """Display suggestions for general and specific questions."""
    st.markdown("### Need some ideas? Try these:")
    
    st.markdown("###### General Questions")
    display_suggestion_buttons(GENERAL_QUESTIONS, key_prefix="general")


def display_suggestion_buttons(questions, key_prefix):
    # Render buttons
    for i, question in enumerate(questions):
        if st.button(question, key=f"{key_prefix}_question_{i}"):
            add_user_input(question)
            st.session_state.suggestions_visible = False
            st.rerun()


def display_chat_history():
    """Render chat history with appropriate roles and icons."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar=user_icon):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar=assistant_icon):
                st.markdown(message["content"])


def handle_user_input():
    """Handle manual user input from the chat box."""
    user_input = st.chat_input("Ask me anything about your health based on your data…")
    if user_input:
        add_user_input(user_input)


def add_user_input(user_input):
    """Add user input to chat history and generate a response."""
    # Append user input to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=user_icon):
        st.markdown(user_input)

    # Generate and display assistant response
    generate_assistant_response(user_input)


def generate_assistant_response(user_input):
    """Generate assistant response by querying the backend."""
    with st.chat_message("assistant", avatar=assistant_icon):
        # Call the backend API
        response = requests.post(BACKEND_URL, json={"question": user_input})

        if response.status_code == 200:
            response_data = response.json()

            # Handle response output
            assistant_response = response_data.get("response", "I'm sorry, I couldn't retrieve the information.")
            if isinstance(assistant_response, dict):
                assistant_response = assistant_response.get("output", "Sorry, I couldn't process that.")

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.markdown(assistant_response)
        else:
            st.markdown("Error: Unable to get response from the server.")


if __name__ == "__main__":
    st.set_page_config(page_title="Chat with Tershine+GPT", layout="wide")
    top_bar = """
    <div style="display: flex; align-items: center; gap: 5px; margin-bottom: 20px;">
        <h1 style="margin: 0;">Chat with Tershine+AgentAI</h1>
    </div>

    """

    # Render the SVG in Streamlit
    st.markdown(top_bar, unsafe_allow_html=True)
    st.markdown("### Tjingelongan och välkomna")
    st.markdown("Welcome to your personal AI assistant, Tershine+AgentAI. May the gloss be with you!")
    main()