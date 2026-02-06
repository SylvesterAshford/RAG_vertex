import streamlit as st
from google import genai
from google.genai import types

# ==================================================
# 1. SETUP & AUTHENTICATION
# ==================================================
st.set_page_config(
    page_title="Gemini Co-Engineer",
    page_icon="🤖",
    layout="centered"
)

# Your API Key from Google AI Studio
API_KEY = "AQ.Ab8RN6K7redysYm8EwTcervBSTriRWFLsXTpDtK5DuAZwFm0zw"

# Initialize the Client for the Gemini Developer API (vertexai=False by default)
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.0-flash"

# ==================================================
# 2. SYSTEM INSTRUCTIONS
# ==================================================
SYSTEM_PROMPT = """
Role: Guided Co-Engineering Coach (Agri Venture Studio).
Mission: Help farmers achieve stable yield and predictable income.
Style: Sharp, peer-to-peer, collaborative. 
Always end with a 'Pivot Question' to keep the conversation moving.
"""

# ==================================================
# 3. CHAT HISTORY & UI
# ==================================================
st.title("Oyster Mushroom Co-Engineer 🍄")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask your farming coach..."):
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing strategy..."):
            try:
                # Format history for the SDK
                history_parts = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])]
                    ) for m in st.session_state.messages[:-1]
                ]

                # Start chat with system instruction and history
                chat = client.chats.create(
                    model=MODEL_ID,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7
                    ),
                    history=history_parts
                )
                
                response = chat.send_message(prompt)
                full_response = response.text
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Chat Error: {e}")