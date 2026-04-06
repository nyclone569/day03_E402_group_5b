import streamlit as st
import os
from dotenv import load_dotenv

from src.core.gemini_provider import GeminiProvider
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
    agent = ReActAgent(llm=llm, tools=TOOLS, max_steps=5)

    with st.chat_message("assistant"):
        with st.spinner("Agent đang suy nghĩ..."):
            try:
                # Run the ReAct Loop
                result = agent.run(prompt)
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                error_msg = f"Đã xảy ra lỗi hệ thống: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
