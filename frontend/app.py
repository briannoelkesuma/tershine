import streamlit as st
import requests  # Import the requests library

# Custom icons for user and assistant
user_icon = "https://cdn-icons-png.flaticon.com/512/6897/6897018.png"
assistant_icon = "https://cdn.handshake.fi/images/autodudese/produktbilder/tershine/logo-tershine.png"

# Suggestions for users in English and Swedish
SUGGESTIONS_EN = [
    "Which degreaser should I use?",
    "How do I clean my car dashboard?",
    "Recommend me the best wax for my car?",
    "How do I clean my car after snow?"
]

SUGGESTIONS_SE = [
    "Vilken avfettningsmedel ska jag använda?",
    "Hur rengör jag bilens instrumentbräda?",
    "Vilket är det bästa vaxet för min bil?",
    "Hur rengör jag min bil efter snö?"
]

# Backend URL
BACKEND_URL = "http://localhost:8000/query/"  # Replace with your backend URL


def main():
    initialise_chat()
    display_language_toggle()
    display_dynamic_content()
    display_suggestions()
    display_chat_history()
    handle_user_input()


def initialise_chat():
    """Initialize the chat history, language, and suggestions visibility."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "suggestions_visible" not in st.session_state:
        st.session_state.suggestions_visible = True

    if "language" not in st.session_state:
        st.session_state.language = "EN"  # Default language is English


def display_language_toggle():
    """Display a language toggle button."""
    st.sidebar.title("Settings")
    language = st.sidebar.radio("Select Language / Välj språk:", ["EN", "SE"])
    st.session_state.language = language

def display_dynamic_content():
    """Display dynamic content based on the selected language."""
    if st.session_state.language == "EN":
        st.title("Chat with Tershine Assistant 🧼")
        st.subheader("Your expert car care assistant")
        st.markdown(
            """
            Welcome to the Tershine Assistant! 🚗  
            Ask anything about car cleaning, maintenance, or Tershine products.
            """
        )
    else:  # Swedish content
        st.title("Chatta med Tershine Assistent 🧼")
        st.subheader("Din expert på bilvård")
        st.markdown(
            """
            Välkommen till Tershine Assistent! 🚗  
            Fråga vad som helst om bilrengöring, underhåll eller Tershine-produkter.
            """
        )

def display_suggestions():
    """Display suggested prompts above the chat input."""
    suggestions = SUGGESTIONS_EN if st.session_state.language == "EN" else SUGGESTIONS_SE

    if st.session_state.suggestions_visible and not st.session_state.chat_history:
        st.subheader("Need some ideas? Try these:" if st.session_state.language == "EN" else "Behöver du tips? Prova dessa:")
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key=f"suggestion_{i}"):
                # Treat clicked suggestion as user input
                add_user_input(suggestion)
                st.session_state.suggestions_visible = False  # Hide suggestions
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
    placeholder_text = "Ask me anything about car care :)" if st.session_state.language == "EN" else "Fråga mig vad som helst om bilvård :)"
    user_input = st.chat_input(placeholder_text)
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
        response = requests.post(BACKEND_URL, json={"question": user_input, "language": st.session_state.language})

        if response.status_code == 200:
            response_data = response.json()

            # Handle response output
            assistant_response = response_data.get("response", "I'm sorry, I couldn't retrieve the information." if st.session_state.language == "EN" else "Jag är ledsen, jag kunde inte hämta informationen.")
            if isinstance(assistant_response, dict):
                assistant_response = assistant_response.get("output", "Sorry, I couldn't process that." if st.session_state.language == "EN" else "Tyvärr kunde jag inte behandla det.")

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.markdown(assistant_response)
        else:
            error_message = "Error: Unable to get response from the server." if st.session_state.language == "EN" else "Fel: Kunde inte få svar från servern."
            st.markdown(error_message)


if __name__ == "__main__":
    st.set_page_config(page_title="Tershine Chatbot", page_icon="🧼")
    main()