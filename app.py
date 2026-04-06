import streamlit as st
import os
from dotenv import load_dotenv

from src.core.gemini_provider import GeminiProvider
from src.core.local_provider import LocalProvider
from src.agent.agent import ReActAgent
from src.agent.tools import TOOLS

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

st.set_page_config(page_title="VNStock AI Agent", page_icon="📈", layout="wide")

st.title("📈 VNStock ReAct Agent")
st.markdown("Hệ thống Hỏi đáp Chứng khoán thông minh sử dụng Gemma-4 và LangChain logic (ReAct).")

# Lấy API Key ngầm từ biến môi trường
api_key = os.getenv("GEMINI_API_KEY")

# Setup Sidebar Config
with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_name = st.text_input("Tên Model", value="gemma-4-31b-it") # Or fallback to gemini-1.5-flash
    
    if not api_key:
        st.error("Lỗi: Chưa cấu hình GEMINI_API_KEY trong file .env!")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "charts" in message and message["charts"]:
            for fig in message["charts"]:
                st.plotly_chart(fig)

# Chat Input
if prompt := st.chat_input("Hỏi tôi về giá mã chứng khoán (VD: Giá FPT)"):
    if not api_key:
        st.error("Lỗi: Chưa có API Key!")
        st.stop()
        
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Instantiate the Backend Agent
    llm = GeminiProvider(model_name=model_name, api_key=api_key)
    # llm = LocalProvider(model_path="./models/gemma-3-4b-it.gguf")
    agent = ReActAgent(llm=llm, tools=TOOLS, max_steps=5)

    with st.chat_message("assistant"):
        with st.spinner("Agent đang suy nghĩ..."):
            try:
                import time
                start_time = time.time()
                # Run the ReAct Loop
                result = agent.run(prompt)
                latency = time.time() - start_time
                steps = getattr(agent, 'current_steps', 0)
                
                meta_str = f"⏱️ Thời gian xử lý: {latency:.2f}s | 🔄 Số bước suy luận (ReAct Steps): {steps}"
                
                st.markdown(result)
                
                # Fetch any charts generated during the agent run
                charts = st.session_state.pop("temp_charts", [])
                for fig in charts:
                    st.plotly_chart(fig)
                    
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result,
                    "charts": charts
                })
            except Exception as e:
                error_msg = f"Đã xảy ra lỗi hệ thống: {str(e)}"
                st.error(error_msg)
                
                charts = st.session_state.pop("temp_charts", [])
                for fig in charts:
                    st.plotly_chart(fig)
                    
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "charts": charts
                })
