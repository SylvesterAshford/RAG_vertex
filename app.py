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

# → Replace this with your Google AI Studio API Key
API_KEY = "AQ.Ab8RN6K7redysYm8EwTcervBSTriRWFLsXTpDtK5DuAZwFm0zw"

# Initialize the client (ensure you installed google‑genai package)
client = genai.Client(api_key=API_KEY)

# Use a suitable Gemini model
MODEL_ID = "gemini-2.0-flash"

# 🤖 System prompt for personality and instructions
SYSTEM_PROMPT = """
Role: Guided Co-Engineering Coach.
Style: Sharp, peer-to-peer, collaborative. 
Mission: Help farmers with oyster mushroom business strategy.
Rule: Always end every response with a focused 'Pivot Question'.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# 2. CHAT INTERFACE
# ==================================================
st.title("Oyster Mushroom Co-Engineer 🍄")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
if user_input := st.chat_input("How can we scale your mushroom yield?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build conversation text
                conversation = SYSTEM_PROMPT + "\n\n"
                for m in st.session_state.messages:
                    role = "User:" if m["role"] == "user" else "Assistant:"
                    conversation += f"{role} {m['content']}\n"

                # Call the Gemini API via generate_content
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=500
                    ),
                )

                assistant_text = response.text

                # Show and save
                st.markdown(assistant_text)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

            except Exception as e:
                st.error(f"Chat Error: {e}")
