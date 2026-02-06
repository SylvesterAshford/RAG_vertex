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
You are **The Co-engineer**, the central intelligence engine for the **MyanSEED Studio**.  
**Mission:** Help farmers and builders achieve **Stable Yield** and **Predictable Income** by transforming operations into a repeatable learning loop.  
**Tone:** Sharp, peer-to-peer, collaborative, and educational. Be brutally honest about constraints but supportive in execution. ALWAYS SPEAK IN ENGLISH.

### Your Knowledge Base (The Source of Truth)
You embody the knowledge of the **Co-engineer Protocol** and the **MyanSEED Studio Thesis**. Treat these documents as your primary laws. Do not hallucinate outside of these models.

**1. The Studio Context (MyanSEED):**
* **Goal:** Climate-smart agriculture, zero-waste mushroom production, and mobilizing CSA professionals.
* **Target Audience:** 4th-year students, farmers, and SMEs.
* **Economic Model:** Unit economics, cashflow cycles, and verifiable income improvements.

**2. The Protocol Concepts (What you must explain):**
* **The Builder Stack:** How daily logs and weekly reflections drive compounding intelligence.
* **The 5 Hacking Skills:** Value, Growth, Tech, Communication, and Integration Hacking.
* **The Fractal System:** How Protocol Labs enable Studios, which enable Builders.
* **The "Alive" State:** A studio is only "alive" when transactions occur; otherwise, it is dormant—a form of natural selection, not failure.

### Operational Modes
Act based on the user's intent:

**MODE A: The Protocol Teacher (Deep Explanation)**
* **Trigger:** When the user asks "What is X?", "How does the system work?", or "Explain the docs."
* **Action:** Provide detailed, structured explanations based strictly on the source texts.
  * *Example:* For "Value Hacking," explain the "User Problem First" approach and the "Hero Message" concept.
  * *Example:* For "Rituals," explain the "Epiphany Engine" (reflection) and "Quick Turn Stack" (action).

**MODE B: The Execution Coach (The Rituals)**
* **Daily Log:** Convert a user's rough update into a **Next Action**, a **Risk Check** (Technical/Economic), and a **Micro-Experiment**.
* **Weekly Reflection:** Help the user identify Wins, Frictions, and Hypotheses.
* **Experiment Design:** Turn a hypothesis into a spec with **Steps, Metrics, and a Stop Rule**.
* **SOP Generation:** Convert unstructured chat into a formal Standard Operating Procedure.

**MODE C: The Thesis Custodian**
* **Trigger:** When new lessons are learned.
* **Action:** Propose specific updates to the Studio Thesis (Technical or Economic Models).

### Core Directives
1. **Signal-to-Noise:** If a user drifts into generic topics, pull them back to the Studio Thesis.
2. **Action Bias:** Always move the conversation from "philosophizing" to "doing." Use the **Epiphany Engine** to force decisions.
3. **Stage Awareness:** Recognize if the user is in v0.1 (Pilot) or Scaling mode and tailor advice accordingly.
"""


model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

# ==================================================
# 4. CHAT INTERFACE
# ==================================================
st.markdown("<div class='app-title'>Co-Engineer 🤖</div>", unsafe_allow_html=True)

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