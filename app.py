import os
import json
import streamlit as st
import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

# =========================
# CONFIGURATION
# =========================
PROJECT_ID = "gen-lang-client-0938066012"
LOCATION = "us-west1"

RAG_CORPUS_NAME = (
    "projects/gen-lang-client-0938066012/"
    "locations/us-west1/"
    "ragCorpora/6917529027641081856"
)

TOP_K = 3
VECTOR_DISTANCE_THRESHOLD = 0.5

# =========================
# STREAMLIT CLOUD SECRET AUTH
# =========================
# Load service account JSON from secrets
sa_json = st.secrets["gcp"]["service_account"]
sa_dict = json.loads(sa_json)

# Save to a temporary file (required by Vertex AI)
key_path = "/tmp/vertexai_sa.json"
with open(key_path, "w") as f:
    json.dump(sa_dict, f)

# Set environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Force project
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

# =========================
# INIT VERTEX AI
# =========================
vertexai.init(project=PROJECT_ID, location=LOCATION)

# =========================
# RAG CONFIG
# =========================
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=TOP_K,
    filter=rag.Filter(vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD),
)

# =========================
# RAG TOOL
# =========================
rag_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS_NAME)],
            rag_retrieval_config=rag_retrieval_config,
        )
    )
)

# =========================
# GEMINI MODEL
# =========================
model = GenerativeModel(
    model_name="gemini-2.0-flash-001",
    tools=[rag_tool],
)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Vertex AI RAG Chatbot", layout="centered")

st.title("🤖 Vertex AI RAG Chatbot")
st.caption("Powered by Gemini + Vertex AI RAG")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking with RAG..."):
            try:
                response = model.generate_content(user_input)
                answer = response.text
            except Exception as e:
                answer = f"⚠️ Error generating response: {e}"

            st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
