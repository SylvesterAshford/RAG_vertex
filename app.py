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
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# CUSTOM CSS (CHATGPT-LIKE UI)
# ==================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 6rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
}

.chat-container {
    max-width: 900px;
    margin: auto;
}

.user-bubble {
    background: #2b2b2b;
    color: white;
    padding: 14px 18px;
    border-radius: 16px;
    margin: 10px 0;
    max-width: 80%;
    float: right;
    clear: both;
}

.ai-bubble {
    background: #f3f4f6;
    color: #111;
    padding: 14px 18px;
    border-radius: 16px;
    margin: 10px 0;
    max-width: 80%;
    float: left;
    clear: both;
}

.stChatInput {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: white;
    padding: 1rem;
    border-top: 1px solid #ddd;
}

.app-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("<div class='app-title'>Olyster Mushroom Business Co-Engineer 🤖</div>", unsafe_allow_html=True)

# ==================================================
# 2. CONFIGURATION (HARDCODED CORPUS)
# ==================================================

PROJECT_ID = "gen-lang-client-0938066012"
LOCATION = "asia-southeast1"

# 🔥 HARDCODED NEW RAG CORPUS
CORPUS_ID = "projects/gen-lang-client-0938066012/locations/asia-southeast1/ragCorpora/6917529027641081856"

# ==================================================
# 3. AUTHENTICATION (STREAMLIT CLOUD READY)
# ==================================================
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
# 4. INITIALIZE RAG TOOL
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
Mission Anchor:
You operate inside the MyanSEED Studio.

Your primary objective is to:
- help farmers achieve stable yield and predictable income
- help MyanSEED produce scalable, real entrepreneurship outcomes

CORE BEHAVIOR:
1. Do not ask generic questions like "What can I do for you?".
2. Start from farmer needs.
3. Identify real operational barriers.
4. Propose low-risk, repeatable actions.
5. Connect actions to measurable outcomes.
6. Every response must end with a Pivot Question.
"""

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

# ==================================================
# 5. STATE
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# 6. CHAT DISPLAY (CHATGPT STYLE)
# ==================================================
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{m['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-bubble'>{m['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# 7. INPUT
# ==================================================
prompt = st.chat_input("Ask your Co-Engineer coach...")

if prompt:
    # store user msg
    st.session_state.messages.append({"role": "user", "content": prompt})

    history = [
        Content(
            role="user" if m["role"] == "user" else "model",
            parts=[Part.from_text(m["content"])]
        )
        for m in st.session_state.messages[:-1]
    ]

    chat = model.start_chat(history=history)
    response = chat.send_message(prompt)

    # store ai msg
    st.session_state.messages.append({"role": "assistant", "content": response.text})

    st.rerun()
