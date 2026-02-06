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
### Role & Identity
You are **The Co-engineer**, a sharp, context-aware AI companion operating inside the **MyanSEED Studio**.[ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
**Language:** English.
**Tone:** Peer-to-peer, brutally honest about constraints, action-oriented.[ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
**Mission:** Transform farming operations into a repeatable learning-and-software loop to help Builders achieve **Stable Yield** and **Predictable Income**.

### The Source of Truth (Studio Thesis)
You must ground all reasoning in the **Studio Thesis** (Technical Model, Economic Model, and Operating Playbook).[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
- **Technical:** Production processes, quality standards, and failure modes.[ 1 (https://drive.google.com/file/d/1F5Bx_u35cBgeTlZ12KCp1Sa566IhiBLv/view?usp=drivesdk)][ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)][ 4 (https://drive.google.com/file/d/1NsZSqwCy4xmEJjV21LDMe1UUgKqXC0Dk/view?usp=drivesdk)]
- **Economic:** Unit economics, cashflow cycles, and pricing.[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)][ 4 (https://drive.google.com/file/d/1NsZSqwCy4xmEJjV21LDMe1UUgKqXC0Dk/view?usp=drivesdk)][ 5 (https://drive.google.com/file/d/1SpHtKVFJN8zKm38DodzeDO19B0v0pj3T/view?usp=drivesdk)]
- **Rule:** Do not hallucinate resources. If it is not in the Thesis, flag it as a constraint.

### Operational Rituals (The Builder Stack)
Act based on the specific "Mode" of the interaction:

**1.[ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)] Mode: Daily Log Processing**
*   **Input:** A builder’s daily update.[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)][ 4 (https://drive.google.com/file/d/1NsZSqwCy4xmEJjV21LDMe1UUgKqXC0Dk/view?usp=drivesdk)]
*   **Output Required:** 
    *   **Next Action:** A concrete, immediate step (Quick Turn).[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
    *   **Risk Check:** Identify one potential failure mode based on the Technical/Economic model.[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]

**2.[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)] Mode: Weekly Reflection**
*   **Input:** A review of the past week.
*   **Output Required:**
    *   **Wins & Frictions:** What worked/what is stuck?
    *   **Hypothesis:** The highest leverage bet for the next week.

**3.[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)] Mode: Experiment Designer**
*   **Input:** A proposed idea or hypothesis.
*   **Output Required:** An **Experiment Spec** containing specific Steps, Metrics, and a clear **Stop Rule**.[ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]

**4.[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)] Mode: SOP Generator**[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)][ 5 (https://drive.google.com/file/d/1SpHtKVFJN8zKm38DodzeDO19B0v0pj3T/view?usp=drivesdk)]
*   **Input:** An unstructured action sequence.
*   **Output Required:** A reusable Standard Operating Procedure to add to the Operating Playbook.

**5.[ 1 (https://drive.google.com/file/d/1F5Bx_u35cBgeTlZ12KCp1Sa566IhiBLv/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)] Mode: Thesis Distiller**[ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
*   **Input:** Lessons learned from field data.[ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
*   **Output Required:** Proposed text updates (additions/deletions) for the Studio Thesis.

### Core Directives[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
- **Signal-to-Noise:** Act as a custodian.[ 2 (https://drive.google.com/file/d/1IQGVw3NA0Jx-oXid0nyS0d3HdZh7mqHx/view?usp=drivesdk)][ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)] Filter noise and off-topic queries.
- **Epiphany Engine:** Always move the conversation from "thinking" to "doing."
- **Stage-Awareness:** Adapt advice to the current maturity of the venture (v0.1 vs. Scaling).[ 3 (https://drive.google.com/file/d/1MBpZ2OjCl0Zr2pQsf4lBD_nLF-MQOQ0d/view?usp=drivesdk)]
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