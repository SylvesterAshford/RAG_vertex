import streamlit as st
import vertexai
import json
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Tool, Content, Part
from vertexai.preview import rag

# ==================================================
# 1. SETUP UI (MUST BE FIRST)
# ==================================================
st.set_page_config(
    page_title="Gemini RAG Co-Engineer",
    page_icon="🍄",
    layout="wide"
)

# ==================================================
# 2. RESPONSIVE CUSTOM CSS
# ==================================================
st.markdown("""
<style>
    /* Main container adjustment */
    .main .block-container {
        max-width: 100%;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Chat Area */
    .chat-container {
        width: 100%;
        max-width: 800px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-bottom: 120px; /* Space for the input box */
    }

    /* Modern Bubbles */
    .bubble {
        padding: 14px 18px;
        border-radius: 18px;
        font-size: 16px;
        line-height: 1.5;
        max-width: 85%;
        word-wrap: break-word;
        font-family: 'Inter', sans-serif;
    }

    .user-bubble {
        background-color: #2b2b2b;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
    }

    .ai-bubble {
        background-color: #f1f3f4;
        color: #1a1a1a;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
        border: 1px solid #e0e0e0;
    }

    .app-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        color: #1a1a1a;
    }

    /* Fix Streamlit's default chat input position */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        z-index: 1000;
    }

    /* Mobile adjustments */
    @media (max-width: 640px) {
        .bubble { max-width: 95%; font-size: 14px; }
        .app-title { font-size: 1.4rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. CONFIGURATION & AUTH
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
# 4. MODEL & RAG SETUP
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
# 5. UI CONTENT
# ==================================================
st.markdown("<div class='app-title'>Oyster Mushroom Business Co-Engineer 🤖</div>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Container for chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for m in st.session_state.messages:
    role_class = "user-bubble" if m["role"] == "user" else "ai-bubble"
    st.markdown(f"<div class='bubble {role_class}'>{m['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Ask your Co-Engineer coach...")

if prompt:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepare history for Gemini
    history = [
        Content(role="user" if m["role"] == "user" else "model", parts=[Part.from_text(m["content"])])
        for m in st.session_state.messages[:-1]
    ]

    # Generate Response
    chat = model.start_chat(history=history)
    with st.spinner("Analyzing data..."):
        response = chat.send_message(prompt)
    
    # Append AI Message
    st.session_state.messages.append({"role": "assistant", "content": response.text})
    
    # Refresh to show new messages
    st.rerun()