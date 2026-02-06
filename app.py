import streamlit as st
from google import genai
from google.genai import types

# ==================================================
# 1. SETUP & AUTHENTICATION
# ==================================================
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="centered")

# Your provided API Key
API_KEY = "AQ.Ab8RN6K7redysYm8EwTcervBSTriRWFLsXTpDtK5DuAZwFm0zw"

# Initialize the Gemini Client using the SDK you shared
# Note: For API key usage, we don't set vertexai=True
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.0-flash"

# ==================================================
# 2. CHAT HISTORY MANAGEMENT
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# 3. CHATGPT STYLE UI
# ==================================================
st.title("Gemini Pro Assistant 🤖")

# Display previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
if prompt := st.chat_input("What is on your mind?"):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Gemini is thinking..."):
            try:
                # Convert session history to SDK format
                history = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])]
                    ) for m in st.session_state.messages[:-1]
                ]

                # Start chat and send message
                chat = client.chats.create(model=MODEL_ID, history=history)
                response = chat.send_message(prompt)
                
                # Display response
                full_response = response.text
                st.markdown(full_response)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")