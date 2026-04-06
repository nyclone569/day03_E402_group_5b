import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import os
import re
from dotenv import load_dotenv

load_dotenv()

# --- CẤU HÌNH ---
DEFAULT_DEMO_MODE = os.getenv("DEMO_MODE", "mock").strip().lower()
# Sử dụng API Key từ biến môi trường
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Chưa thấy GEMINI_API_KEY hoặc GOOGLE_API_KEY trong .env!")

# Khởi tạo mô hình
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it", # Đã chuyển sang gemma theo yêu cầu
    temperature=0, 
    google_api_key=api_key
)

# --- 1. ĐỊNH NGHĨA TOOLS ---
def get_stock_price(symbol: str) -> str:
    """Lấy giá chứng khoán VN."""
    mock_data = {"HPG": 30500, "VNM": 68000, "SSI": 37500}
    price = mock_data.get(symbol.upper(), "không tìm thấy")
    return f"Giá hiện tại của {symbol.upper()} là {price:,} VND."

def create_stock_chart(symbol: str) -> str:
    """Tạo biểu đồ kỹ thuật."""
    st.session_state.show_chart = symbol.upper()
    return f"Đã khởi tạo biểu đồ cho mã {symbol.upper()}. Hãy nhìn vào phần hiển thị biểu đồ bên dưới."

def extract_symbol(text: str) -> str:
    """Trích xuất mã cổ phiếu 3 ký tự, ưu tiên mã viết hoa."""
    candidates = re.findall(r"\b[A-Z]{3}\b", text.upper())
    return candidates[0] if candidates else ""

def process_mock_request(user_input: str) -> str:
    """Luồng demo local để chạy khi chưa có API hoặc API lỗi."""
    symbol = extract_symbol(user_input)
    lowered = user_input.lower()

    if not symbol:
        return "Bạn hãy nhập mã cổ phiếu 3 ký tự (ví dụ: HPG, SSI, FPT)."

    if any(word in lowered for word in ["biểu đồ", "ve", "vẽ", "chart"]):
        tool_result = create_stock_chart(symbol)
        return f"[MOCK] Action: Vẽ biểu đồ {symbol}\nKết quả: {tool_result}"

    tool_result = get_stock_price(symbol)
    return f"[MOCK] Action: Lấy giá {symbol}\nKết quả: {tool_result}"

tools = [
    Tool(name="GetPrice", func=get_stock_price, description="Lấy giá hiện tại của cổ phiếu VN. Tham số đầu vào là mã cổ phiếu 3 chữ cái (VD: HPG)."),
    Tool(name="CreateChart", func=create_stock_chart, description="Vẽ biểu đồ lịch sử giá cho cổ phiếu. Tham số đầu vào là mã cổ phiếu 3 chữ cái (VD: HPG).")
]

# --- 2. PROMPT CHUẨN HÓA VIỆT NAM ---
system_prompt = """
Bạn là chuyên gia chứng khoán VN.
Quy tắc:
1. Trả lời bằng tiếng Việt.
2. Giá tiền đơn vị VND.
3. Nếu cần vẽ biểu đồ, hãy gọi tool CreateChart.
4. Trả lời trực tiếp và súc tích nếu không cần thêm công cụ.
"""

# --- 3. KHỞI TẠO AGENT ---
# LangGraph use create_react_agent to natively handle loops and tool calling
agent_executor = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=system_prompt
)

# --- 4. GIAO DIỆN UI (STREAMLIT) ---
st.set_page_config(page_title="VNStock AI (LangChain ReAct)", layout="wide")
st.title("🚀 VNStock ReAct Agent (Gemini + LangGraph)")

use_mock_mode = st.sidebar.toggle(
    "Mock mode (không gọi Gemini API)",
    value=(DEFAULT_DEMO_MODE != "api")
)

if use_mock_mode:
    st.sidebar.info("Đang chạy ở chế độ mock để demo nhanh.")
else:
    st.sidebar.info("Đang chạy qua Gemini API.")

if "show_chart" not in st.session_state: 
    st.session_state.show_chart = None

col1, col2 = st.columns([1, 1])

with col1:
    user_input = st.text_input("Nhập câu hỏi (VD: Giá HPG bao nhiêu và vẽ biểu đồ cho tôi):")
    if user_input:
        if use_mock_mode:
            with st.spinner("Đang xử lý Mock local..."):
                response = process_mock_request(user_input)
                st.write(response)
        else:
            with st.spinner("Đang phân tích (Bao gồm suy nghĩ)..."):
                try:
                    response = agent_executor.invoke({"messages": [("user", user_input)]})
                    # Trích xuất thought process nếu có
                    output_message = response["messages"][-1].content
                    
                    # Trong LangChain Google GenAI, content đôi khi được trả về dưới dạng list các block 
                    # (chứa cả thinking & text)
                    if isinstance(output_message, list):
                        thoughts = [block['thinking'] for block in output_message if block.get('type') == 'thinking']
                        text_ans = "".join(block['text'] for block in output_message if block.get('type') == 'text')
                        
                        if thoughts:
                            with st.expander("🤔 Tiến trình suy luận (Thinking)"):
                                st.write("\n".join(thoughts))
                        
                        st.write(text_ans)
                    else:
                        st.write(output_message)
                except Exception as e:
                    # Fallback sang mock data nếu API lỗi
                    fallback = process_mock_request(user_input)
                    st.error(f"Lỗi gọi API: {e}\n\nĐã chuyển tự động sang chế độ Mock:\n\n{fallback}")

with col2:
    if st.session_state.show_chart:
        st.subheader(f"Biểu đồ mã: {st.session_state.show_chart}")
        # Demo Plotly Chart
        df = pd.DataFrame({'Ngày': ['1/4', '2/4', '3/4'], 'Giá': [29, 30, 31]})
        fig = go.Figure(data=[go.Candlestick(x=df['Ngày'], open=[29,29,30], high=[31,31,32], low=[28,28,29], close=[30,31,31])])
        st.plotly_chart(fig)
