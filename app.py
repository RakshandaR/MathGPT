import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler

# The modern 2026 way to import agents
from langchain.agents import create_agent 

## Streamlit Setup
st.set_page_config(page_title="MathGPT", page_icon="🧮")
st.title("MathGPT: 2026 Edition")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API Key to continue.")
    st.stop()

# 1. Initialize modern LLM
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# 2. Define Tools using the standard @tool decorator
@tool
def wikipedia_search(query: str):
    """Search Wikipedia for math constants, history, or formulas."""
    return WikipediaAPIWrapper().run(query)

@tool
def math_solver(problem: str):
    """Solve arithmetic and algebraic problems. Input: '5 + 5 * 10'."""
    # We use the LLM as a logical calculator
    return llm.invoke(f"Solve this math problem: {problem}. Return only the result.").content

tools = [wikipedia_search, math_solver]

# 3. Initialize the Agent (This replaces AgentExecutor)
# The system_prompt guides the model to give a clear final answer.
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a precise math assistant. Always provide a clear 'Final Answer' at the end of your logic."
)

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm MathGPT. What can I calculate for you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # The Streamlit callback now lives in langchain_community.callbacks
        st_callback = StreamlitCallbackHandler(st.container())
        
        # We invoke the agent directly. In v1.x, it returns the final state.
        result = agent.invoke(
            {"messages": [("user", user_query)]},
            config={"callbacks": [st_callback]}
        )
        
        # The result is a dictionary containing a list of messages.
        # We take the content of the very last message in the sequence.
        final_answer = result["messages"][-1].content
        st.write(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})