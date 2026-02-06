import streamlit as st
from google import genai
from google.genai import types

# ==================================================
# 1. SETUP UI
# ==================================================
st.set_page_config(
    page_title="Gemini Co-Engineer",
    page_icon="🤖",
    layout="centered"
)

# Replace with your API key
API_KEY = "AQ.Ab8RN6K7redysYm8EwTcervBSTriRWFLsXTpDtK5DuAZwFm0zw"

# FIX: We DO NOT use vertexai=True here because you are using an AI Studio API Key.
client = genai.Client("AQ.Ab8RN6K7redysYm8EwTcervBSTriRWFLsXTpDtK5DuAZwFm0zw")
MODEL_ID = "gemini-2.0-flash"

# ==================================================
# 2. CHAT HISTORY & SYSTEM PROMPT
# ==================================================
SYSTEM_PROMPT = """
Role: Guided Co-Engineering Coach.
Style: Sharp, peer-to-peer, collaborative. 
Mission: Help farmers with oyster mushroom business strategy.
Rule: Always end every response with a focused 'Pivot Question'.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# 3. CHAT INTERFACE (ChatGPT Style)
# ==================================================
st.title("Oyster Mushroom Co-Engineer 🍄")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("How can we scale your mushroom yield?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Calculating..."):
            try:
                # Format previous messages for the SDK
                history_for_api = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])]
                    ) for m in st.session_state.messages[:-1]
                ]

                # Create the chat session
                chat = client.chats.create(
                    model=MODEL_ID,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7
                    ),
                    history=history_for_api
                )
                
                response = chat.send_message(prompt)
                
                # Show response and save to history
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

            except Exception as e:
                # This will now catch any other issues
                st.error(f"Chat Error: {e}")