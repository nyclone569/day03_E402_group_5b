import os
import re
import time
import unicodedata
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.agent.tools import execute_tool_logic

class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop
    specifically tailored for the VNStock application.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []
        
        # === Trace & Evaluation System ===
        self.trace_log: List[Dict[str, Any]] = []   # Từng bước Thought/Action/Observation
        self.current_steps = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.total_latency_ms = 0
        self.security_flags: List[str] = []          # Ghi lại các cảnh báo bảo mật

    def get_system_prompt(self) -> str:
        """
        System prompt instructing the agent to follow ReAct strictly.
        """
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        return f"""
        Bạn là trợ lý ảo chuyên nghiệp về Chứng khoán Việt Nam (VNStock). Mọi dữ liệu giá cả và biểu đồ PHẢI được lấy thông qua các công cụ. Tuyệt đối KHÔNG ĐƯỢC tự bịa (hallucinate) ra giá cổ phiếu.
        
        Bạn có các công cụ (Tools) sau đây:
        {tool_descriptions}

        Quy tắc BẮT BUỘC (Strict formatting):
        Bạn PHẢI sử dụng định dạng dưới đây cho TỪNG BƯỚC suy luận:
        Thought: [Suy nghĩ của bạn: Tôi cần sử dụng công cụ gì để đáp ứng câu hỏi này?]
        Action: [Tên công cụ, ví dụ: GetPrice(FPT)]
        Observation: [Kết quả trả về từ hệ thống sễ được điền tự động, không tự viết phần này]
        ... (Lặp lại Thought/Action/Observation cho đến khi có đủ thông tin)
        Thought: [Tôi đã có đủ thông tin]
        Final Answer: [Câu trả lời đầy đủ kèm theo timestamp nhận được từ công cụ lấy giá, giữ nguyên format giá trị từ tool]
        """

    def _normalize_vietnamese(self, text: str) -> str:
        """Convert NFD to NFC, and optionally can be used to remove accents."""
        return unicodedata.normalize('NFC', text)

    def _remove_accents(self, text: str) -> str:
        """Removes Vietnamese accents/diacritics for robust keyword matching."""
        nfkd = unicodedata.normalize('NFKD', text)
        no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
        return no_accents.replace('đ', 'd').replace('Đ', 'D')

    def _check_intent(self, user_input: str) -> bool:
        """Guardrail: Strict Intent Check focusing exclusively on 3 usecases."""
        clean_input = self._remove_accents(user_input).upper()
        
        # === Security: Kiểm tra Prompt Injection ===
        injection_patterns = r'(IGNORE PREVIOUS|SYSTEM PROMPT|FORGET INSTRUCTIONS|REVEAL API)'
        if re.search(injection_patterns, clean_input):
            self.security_flags.append(f"⚠️ Prompt Injection detected: '{user_input[:80]}'")
            return False
        
        # 1. Explicit Keyword Blacklist (Phát hiện ngay những thứ cấm)
        forbidden_patterns = r'\b(VANG|FOREX|COIN|CRYPTO|BITCOIN|DAT LENH|MUA|BAN)\b'
        if re.search(forbidden_patterns, clean_input):
            self.security_flags.append(f"🚫 Blocked keyword in: '{user_input[:80]}'")
            return False
            
        # 2. Ngặt nghèo (Whitelist): Dùng LLM đánh giá xem có đúng là 1 trong 3 tác vụ không.
        prompt = f"""
        Nhiệm vụ: Phân loại ý định của câu hỏi.
        Chỉ có đúng 3 loại yêu cầu sau được coi là HỢP LỆ:
        1. Xem/tra cứu giá cổ phiếu.
        2. Tìm ý nghĩa/hỏi mã cổ phiếu của công ty.
        3. Vẽ biểu đồ cổ phiếu.
        
        Câu hỏi: "{user_input}"
        
        Nếu câu hỏi đúng là 1 trong 3 yêu cầu trên, trả lời duy nhất: YES
        Nếu là câu hỏi khác (nói chuyện phiếm, tính toán, thời tiết, luật pháp, v.v.), trả lời duy nhất: NO
        """
        response = self.llm.generate(prompt=prompt)
        self._track_usage(response, "Intent Check")
        decision = response.get("content", "").strip().upper()
        
        if "YES" in decision:
            return True
        return False
    
    def _track_usage(self, response: Dict[str, Any], label: str = ""):
        """Thu thập token usage và latency từ mỗi lần gọi LLM."""
        usage = response.get("usage", {})
        latency = response.get("latency_ms", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total
        self.total_latency_ms += latency
        
        # Ước tính chi phí (Gemini API pricing - có thể thay đổi)
        # Gemma 4 via Gemini API: ~$0.10/1M input, ~$0.30/1M output (ước tính)
        cost = (prompt_tokens * 0.10 / 1_000_000) + (completion_tokens * 0.30 / 1_000_000)
        self.total_cost_usd += cost

    def run(self, user_input: str) -> str:
        """
        ReAct loop logic:
        1. Guardrail / Intent Check -> Unified Fallback
        2. Generate Thought + Action
        3. Parse Action and execute Tool (API Error Handler & Human Escalation)
        4. Check Tool Accuracy (Action Error Handler)
        5. Append Observation and repeat
        """
        # Reset trace cho mỗi lần chạy mới
        self.trace_log = []
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.total_latency_ms = 0
        self.security_flags = []
        run_start = time.time()
        
        # Chuẩn hóa Unicode tiếng Việt (NFC) để model LLM xử lý mượt mà hơn
        user_input = self._normalize_vietnamese(user_input)
        
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        
        # 1. Guardrail (Intent Check)
        self.trace_log.append({"step": 0, "type": "guardrail", "action": "Intent Check", "input": user_input})
        if not self._check_intent(user_input):
            self.trace_log[-1]["result"] = "❌ OUT OF SCOPE"
            logger.log_event("FALLBACK_OUT_OF_SCOPE", {"input": user_input})
            logger.log_event("AGENT_END", {"steps": 0})
            self.current_steps = 0
            return "Xin lỗi, tôi chỉ hỗ trợ tra cứu thông tin tĩnh về Chứng khoán Việt Nam (giá, biểu đồ, mã công ty). Tôi không hỗ trợ đặt lệnh, mua bán, hoặc dự báo vàng/ngoại tệ."
        self.trace_log[-1]["result"] = "✅ IN SCOPE"

        current_prompt = f"User Query: {user_input}\n"
        steps = 0
        
        action_regex = re.compile(r"Action:\s*([A-Za-z0-9_]+)\((.*)\)", re.IGNORECASE)
        final_answer_regex = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
        
        ValidTools = [t['name'] for t in self.tools]
        api_failure_count = 0

        while steps < self.max_steps:
            step_start = time.time()
            
            # Generate LLM response
            result = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())
            self._track_usage(result, f"Step {steps + 1}")
            content = result.get("content", "")
            current_prompt += f"{content}\n"
            
            # Trích xuất Thought từ LLM output
            thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", content, re.DOTALL | re.IGNORECASE)
            thought_text = thought_match.group(1).strip() if thought_match else "(Không rõ)"
            
            step_trace = {
                "step": steps + 1,
                "type": "react_loop",
                "thought": thought_text,
                "llm_raw": content[:500],  # Giới hạn 500 ký tự để debug
                "latency_ms": result.get("latency_ms", 0),
                "tokens": result.get("usage", {}),
            }
            
            # Check for Final Answer
            final_match = final_answer_regex.search(content)
            if final_match:
                step_trace["action"] = "Final Answer"
                step_trace["observation"] = final_match.group(1).strip()[:200]
                self.trace_log.append(step_trace)
                logger.log_event("AGENT_END", {"steps": steps + 1})
                self.current_steps = steps + 1
                return final_match.group(1).strip()
            
            # Parse Thought/Action from result
            action_match = action_regex.search(content)
            if action_match:
                tool_name = action_match.group(1).strip()
                args = action_match.group(2).strip()
                step_trace["action"] = f"{tool_name}({args})"
                
                # Check Action Error (Fallback Action Name)
                if tool_name not in ValidTools:
                    obs_msg = f"❌ Tool `{tool_name}` không tồn tại. Hợp lệ: {', '.join(ValidTools)}."
                    current_prompt += f"Observation: Lỗi sai tên Tool! Tool `{tool_name}` không tồn tại. Vui lòng chỉ sử dụng một trong các tool sau: {', '.join(ValidTools)}.\n"
                    step_trace["observation"] = obs_msg
                    step_trace["status"] = "⚠️ ACTION_ERROR"
                    logger.log_event("ACTION_ERROR", {"wrong_tool": tool_name})
                else:
                    # Execute Tool
                    try:
                        tool_start = time.time()
                        obs = execute_tool_logic(tool_name, args)
                        tool_latency = int((time.time() - tool_start) * 1000)
                        current_prompt += f"Observation: {obs}\n"
                        step_trace["observation"] = obs
                        step_trace["tool_latency_ms"] = tool_latency
                        step_trace["status"] = "✅ SUCCESS"
                        api_failure_count = 0
                    except Exception as e:
                        api_failure_count += 1
                        step_trace["observation"] = f"❌ API Error: {str(e)}"
                        step_trace["status"] = f"🔴 API_ERROR (retry {api_failure_count}/3)"
                        logger.log_event("API_ERROR", {"tool_name": tool_name, "error": str(e), "retry_count": api_failure_count})
                        
                        if api_failure_count >= 3:
                            fallback_msg = "Xin lỗi, Dữ liệu API hiện đang cập nhật chậm hoặc bị bảo trì. Vui lòng liên hệ người thật hoặc thử lại sau."
                            step_trace["status"] = "🚨 HUMAN_ESCALATION"
                            self.trace_log.append(step_trace)
                            logger.log_event("FALLBACK_HUMAN_ESCALATION", {"failures": api_failure_count})
                            logger.log_event("AGENT_END", {"steps": steps + 1})
                            self.current_steps = steps + 1
                            return fallback_msg
                        else:
                            current_prompt += f"Observation: Lỗi API ({str(e)}). Vui lòng Action thử lại.\n"
            else:
                step_trace["action"] = "(Sai định dạng)"
                step_trace["observation"] = "Hệ thống yêu cầu đúng format"
                step_trace["status"] = "⚠️ FORMAT_ERROR"
                current_prompt += "Observation: Vui lòng sử dụng đúng định dạng Action: tool_name(args) hoặc Final Answer: nội dung.\n"
            
            self.trace_log.append(step_trace)
            steps += 1
            
        logger.log_event("AGENT_END", {"steps": steps})
        self.current_steps = steps
        return "Xin lỗi, hệ thống AI đã suy nghĩ quá lâu (vượt quá số bước tối đa) mà không tìm được kết quả."
