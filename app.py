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
    
    st.markdown("---")
    st.markdown("🛠️ **Môi trường Test**")
    model_mode = st.radio("Chế độ Model", ["ReAct Agent (Sử dụng Tool)", "Baseline (Không dùng Tool)"], index=0)
    
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
                
                if "Baseline" in model_mode:
                    # Chế độ Baseline: Hỏi thẳng LLM không bảo vệ bằng ReAct loop hay Tools
                    baseline_prompt = f"Người dùng hỏi: {prompt}\nHãy đóng vai trợ lý AI trả lời dựa trên bộ nhớ gốc (Không có công cụ hỗ trợ ngoài)."
                    response = llm.generate(baseline_prompt)
                    result = response.get("content", "Lỗi: Không có phản hồi từ LLM.")
                    latency = time.time() - start_time
                    usage = response.get("usage", {})
                    meta_str = f"⏱️ Thời gian xử lý: {latency:.2f}s | 🧠 Chế độ: Baseline (Dùng não tự đoán)"
                    
                    st.markdown(result)
                    st.caption(meta_str)
                    
                    # Baseline trace đơn giản
                    with st.expander("📊 Chi tiết Baseline"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("⏱️ Latency", f"{latency:.2f}s")
                        col2.metric("🔢 Tokens", f"{usage.get('total_tokens', 0):,}")
                        col3.metric("🛡️ Tools Used", "0")
                        st.info("ℹ️ Baseline gọi LLM 1 lần duy nhất, không qua Guardrail hay Tool.")
                    
                    st.session_state.messages.append({"role": "assistant", "content": result, "meta": meta_str})
                else:
                    # Run the ReAct Loop
                    result = agent.run(prompt)
                    latency = time.time() - start_time
                    steps = getattr(agent, 'current_steps', 0)
                    meta_str = f"⏱️ Thời gian xử lý: {latency:.2f}s | 🔄 Số bước suy luận (ReAct Steps): {steps}"
                    
                    st.markdown(result)
                    st.caption(meta_str)
                    
                    # === TRACE PANEL ===
                    with st.expander("🔍 ReAct Trace (Thought → Action → Observation)"):
                        for trace in agent.trace_log:
                            step_num = trace.get("step", "?")
                            trace_type = trace.get("type", "")
                            
                            if trace_type == "guardrail":
                                st.markdown(f"**Step {step_num}: 🛡️ Guardrail — {trace.get('action', '')}**")
                                st.markdown(f"- Input: `{trace.get('input', '')[:100]}`")
                                st.markdown(f"- Result: {trace.get('result', '')}")
                                st.markdown("---")
                            else:
                                status = trace.get("status", "")
                                st.markdown(f"**Step {step_num}: {status}**")
                                st.markdown(f"- 💭 **Thought**: {trace.get('thought', '—')}")
                                st.markdown(f"- ⚡ **Action**: `{trace.get('action', '—')}`")
                                st.markdown(f"- 👁️ **Observation**: {trace.get('observation', '—')}")
                                
                                tok = trace.get("tokens", {})
                                tool_lat = trace.get("tool_latency_ms", "—")
                                st.caption(f"LLM Latency: {trace.get('latency_ms', 0)}ms | Tool Latency: {tool_lat}ms | Tokens: {tok.get('total_tokens', 0)}")
                                st.markdown("---")
                    
                    # === COST & EVALUATION PANEL ===
                    with st.expander("💰 Cost / Token Usage / Evaluation"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📥 Prompt Tokens", f"{agent.total_prompt_tokens:,}")
                        col2.metric("📤 Completion Tokens", f"{agent.total_completion_tokens:,}")
                        col3.metric("🔢 Total Tokens", f"{agent.total_tokens:,}")
                        col4.metric("💵 Est. Cost", f"${agent.total_cost_usd:.6f}")
                        
                        st.markdown("---")
                        st.markdown("**📈 Evaluation Metrics**")
                        eval_col1, eval_col2, eval_col3 = st.columns(3)
                        eval_col1.metric("🔄 ReAct Steps", steps)
                        eval_col2.metric("⏱️ Total Latency", f"{latency:.2f}s")
                        eval_col3.metric("📡 LLM Calls", len([t for t in agent.trace_log if t.get("type") == "react_loop"]))
                    
                    # === SECURITY PANEL ===
                    if agent.security_flags:
                        with st.expander("🔒 Security Audit Log"):
                            for flag in agent.security_flags:
                                st.warning(flag)
                    


                    charts = st.session_state.pop("temp_charts", [])
                    for fig in charts:
                        st.plotly_chart(fig)
                    st.session_state.messages.append({"role": "assistant", "content": result, "meta": meta_str})
            except Exception as e:
                error_msg = f"Đã xảy ra lỗi hệ thống: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

