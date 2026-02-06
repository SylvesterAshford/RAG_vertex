import streamlit as st
import vertexai
import os
import json
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Tool, Content, Part
from vertexai.preview import rag
from dotenv import load_dotenv

# =========================
# GLOBAL PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Gemini RAG Co‑Engineer",
    page_icon="🤖",
    layout="wide",   # <<< makes it full width like ChatGPT
)

# =========================
# CUSTOM CSS (CHATGPT-LIKE UI)
# =========================
st.markdown("""
<style>
/* Remove Streamlit padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
}

/* Chat container */
.chat-container {
    max-width: 900px;
    margin: auto;
}

/* User bubble */
.user-bubble {
    background: #2b2b2b;
    color: white;
    padding: 14px 18px;
    border-radius: 16px;
    margin: 8px 0;
    max-width: 80%;
    float: right;
    clear: both;
}

/* Assistant bubble */
.ai-bubble {
    background: #f3f4f6;
    color: #111;
    padding: 14px 18px;
    border-radius: 16px;
    margin: 8px 0;
    max-width: 80%;
    float: left;
    clear: both;
}

/* Input box */
.stChatInput {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: white;
    padding: 1rem;
    border-top: 1px solid #ddd;
}

/* Title */
.app-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='app-title'>Olyster Mushroom Business Co‑Engineer 🤖</div>", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
RAW_CORPUS_ID = os.getenv("CORPUS_ID")
CORPUS_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAW_CORPUS_ID}"

# =========================
# AUTH
# =========================
raw_creds = st.secrets["gcp_service_account"]
creds_info = dict(raw_creds) if not isinstance(raw_creds, str) else json.loads(raw_creds)
if "private_key" in creds_info:
    creds_info["private_key"] = creds_info["private_key"].strip().replace("\\n", "\n")

credentials = service_account.Credentials.from_service_account_info(creds_info)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# =========================
# RAG TOOL
# =========================
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
Mission Anchor:
You operate inside the MyanSEED Studio.
"""

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

# =========================
# STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# CHAT DISPLAY
# =========================
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{m['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-bubble'>{m['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# INPUT
# =========================
prompt = st.chat_input("Ask your Co‑Engineer coach...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    history = [
        Content(role="user" if m["role"] == "user" else "model",
                parts=[Part.from_text(m["content"])])
        for m in st.session_state.messages[:-1]
    ]

    chat = model.start_chat(history=history)
    response = chat.send_message(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response.text})

    st.rerun()
