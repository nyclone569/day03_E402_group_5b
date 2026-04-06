import os
import re
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
        
        # 1. Explicit Keyword Blacklist (Phát hiện ngay những thứ cấm)
        forbidden_patterns = r'\b(VANG|FOREX|COIN|CRYPTO|BITCOIN|DAT LENH|MUA|BAN)\b'
        if re.search(forbidden_patterns, clean_input):
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
        decision = response.get("content", "").strip().upper()
        
        if "YES" in decision:
            return True
        return False

    def run(self, user_input: str) -> str:
        """
        ReAct loop logic:
        1. Guardrail / Intent Check -> Unified Fallback
        2. Generate Thought + Action
        3. Parse Action and execute Tool (API Error Handler & Human Escalation)
        4. Check Tool Accuracy (Action Error Handler)
        5. Append Observation and repeat
        """
        # Chuẩn hóa Unicode tiếng Việt (NFC) để model LLM xử lý mượt mà hơn
        user_input = self._normalize_vietnamese(user_input)
        
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        
        # 1. Guardrail (Intent Check)
        if not self._check_intent(user_input):
            logger.log_event("FALLBACK_OUT_OF_SCOPE", {"input": user_input})
            logger.log_event("AGENT_END", {"steps": 0})
            return "Xin lỗi, tôi chỉ hỗ trợ tra cứu thông tin tĩnh về Chứng khoán Việt Nam (giá, biểu đồ, mã công ty). Tôi không hỗ trợ đặt lệnh, mua bán, hoặc dự báo vàng/ngoại tệ."

        current_prompt = f"User Query: {user_input}\n"
        steps = 0
        
        action_regex = re.compile(r"Action:\s*([A-Za-z0-9_]+)\((.*)\)", re.IGNORECASE)
        final_answer_regex = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
        
        ValidTools = [t['name'] for t in self.tools]
        api_failure_count = 0

        while steps < self.max_steps:
            # Generate LLM response
            result = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())
            content = result.get("content", "")
            current_prompt += f"{content}\n"
            
            # Check for Final Answer
            final_match = final_answer_regex.search(content)
            if final_match:
                logger.log_event("AGENT_END", {"steps": steps})
                return final_match.group(1).strip()
            
            # Parse Thought/Action from result
            action_match = action_regex.search(content)
            if action_match:
                tool_name = action_match.group(1).strip()
                args = action_match.group(2).strip()
                
                # Check Action Error (Fallback Action Name)
                # Test Case "Gõ sai Get_Price"
                if tool_name not in ValidTools:
                    # e.g., if tool_name is "Get_Price", advise to use "GetPrice"
                    current_prompt += f"Observation: Lỗi sai tên Tool! Tool `{tool_name}` không tồn tại. Vui lòng chỉ sử dụng một trong các tool sau: {', '.join(ValidTools)}.\n"
                    logger.log_event("ACTION_ERROR", {"wrong_tool": tool_name})
                else:
                    # Execute Tool
                    try:
                        obs = execute_tool_logic(tool_name, args)
                        current_prompt += f"Observation: {obs}\n"
                        api_failure_count = 0 # reset on success
                    except Exception as e:
                        # Fallback API Error Handler
                        api_failure_count += 1
                        logger.log_event("API_ERROR", {"tool_name": tool_name, "error": str(e), "retry_count": api_failure_count})
                        
                        if api_failure_count >= 3:
                            fallback_msg = "Xin lỗi, Dữ liệu API hiện đang cập nhật chậm hoặc bị bảo trì. Vui lòng liên hệ người thật hoặc thử lại sau."
                            logger.log_event("FALLBACK_HUMAN_ESCALATION", {"failures": api_failure_count})
                            logger.log_event("AGENT_END", {"steps": steps})
                            return fallback_msg
                        else:
                            current_prompt += f"Observation: Lỗi API ({str(e)}). Vui lòng Action thử lại.\n"
            else:
                # Agent got confused, force it to correct itself
                current_prompt += "Observation: Vui lòng sử dụng đúng định dạng Action: tool_name(args) hoặc Final Answer: nội dung.\n"
            
            steps += 1
            
        logger.log_event("AGENT_END", {"steps": steps})
        return "Xin lỗi, hệ thống AI đã suy nghĩ quá lâu (vượt quá số bước tối đa) mà không tìm được kết quả."
