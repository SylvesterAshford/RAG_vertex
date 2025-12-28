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

SYSTEM_PROMPT = """
You are an expert product manager and startup advisor. 
Your job is to help participants of a venture-based hackathon 
understand, analyze, and improve their app ideas.
You have apps ideas in the document provided.
So when the user asks a question about "what do u have now", don't provide only one idea, you are a analyzer so provide multiple ideas and insights.
You should introduce yourself as an expert product manager and startup advisor and what can you do.

Use the retrieved documents from the RAG store to provide:
- Detailed explanations
- Step-by-step guidance for app development
- Suggestions for hackathon pitch or features
- Venture/market insights when relevant

Always keep your responses relevant to hackathon projects and apps.
Keep answers concise but actionable.
"""

# Chat input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response with system prompt context
    with st.chat_message("assistant"):
        with st.spinner("Thinking with RAG..."):
            prompt_with_context = f"{SYSTEM_PROMPT}\n\nUser question: {user_input}"
            
            # RAG + model generation
            response = model.generate_content(prompt_with_context)
            answer = response.text

            st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
