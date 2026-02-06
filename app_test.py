import streamlit as st
import vertexai
import json
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Tool, Content, Part
from vertexai.preview import rag

# ==================================================
# 1. SETUP UI (STREAMLIT NATIVE CHAT)
# ==================================================
st.set_page_config(
    page_title="Gemini RAG Co-Engineer",
    page_icon="🍄",
    layout="centered"  # Centered layout feels more like ChatGPT
)

# Custom CSS to refine the look and fix the header
st.markdown("""
<style>
    /* Remove unnecessary padding at the top */
    .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    
    /* Make the title sticky or clean */
    .app-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        color: #1a1a1a;
    }

    /* Style the Chat Input to look cleaner */
    div[data-testid="stChatInput"] {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 2. CONFIGURATION & AUTH
# ==================================================
PROJECT_ID = "gen-lang-client-0938066012"
LOCATION = "us-west1"
CORPUS_ID = "projects/gen-lang-client-0938066012/locations/us-west1/ragCorpora/2305843009213693952"

try:
    raw_creds = st.secrets["gcp"]["service_account"]
    creds_info = json.loads(raw_creds)
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].strip().replace("\\n", "\n")
    
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
except Exception as e:
    st.error(f"❌ Auth Error: {e}")
    st.stop()

# ==================================================
# 3. MODEL & RAG SETUP
# ==================================================
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=CORPUS_ID)],
            similarity_top_k=3,
        ),
    )
)

GUIDED_SYSTEM_PROMPT = """
Role: Guided Co-Engineering Coach (Agri Venture Studio).
Language: ALWAYS respond in English.
Style: sharp, peer-to-peer, collaborative.
Mission Anchor: You operate inside the MyanSEED Studio.
Objective: Help farmers achieve stable yield and predictable income.
"""

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

# ==================================================
# 4. CHAT INTERFACE
# ==================================================
st.markdown("<div class='app-title'>Oyster Mushroom Co-Engineer 🤖</div>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask your Co-Engineer coach..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare history for Gemini
    history = [
        Content(role="user" if m["role"] == "user" else "model", parts=[Part.from_text(m["content"])])
        for m in st.session_state.messages[:-1]
    ]

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat = model.start_chat(history=history)
            response = chat.send_message(prompt)
            st.markdown(response.text)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.text})