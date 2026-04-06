import pytest
from typing import Dict, Any, Optional
from src.agent.agent import ReActAgent
from src.agent.tools import TOOLS
from src.core.llm_provider import LLMProvider
import src.agent.tools as tools_module

class MockLLM(LLMProvider):
    def __init__(self, responses: list):
        super().__init__("mock", "")
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        if self.call_count < len(self.responses):
            ans = self.responses[self.call_count]
            self.call_count += 1
            return {"content": ans}
        return {"content": "Final Answer: Mock empty"}
        
    def stream(self, prompt, system_prompt=None):
        pass

def test_tc1_get_price():
    responses = [
        "Thought: I need to get the price of FPT.\nAction: GetPrice(FPT)",
        "Thought: I got the price.\nFinal Answer: Giá của FPT là 120,000 VND"
    ]
    agent = ReActAgent(llm=MockLLM(responses), tools=TOOLS)
    result = agent.run("Giá FPT hôm nay")
    assert "120,000" in result

def test_tc4_tc5_out_of_scope():
    agent = ReActAgent(llm=MockLLM([]), tools=TOOLS)
    result = agent.run("Dự báo giá vàng thế giới")
    assert "chỉ hỗ trợ" in result.lower() or "không hỗ trợ" in result.lower()
    
    result2 = agent.run("Mua 1000 cổ phiếu VCB")
    assert "chỉ hỗ trợ" in result2.lower() or "không hỗ trợ" in result2.lower()

def test_tc7_action_error_handler():
    responses = [
        "Thought: Let's get the price.\nAction: Get_Price(HPG)", # Wrong action
        "Thought: Ah, I typed it wrong.\nAction: GetPrice(HPG)",
        "Thought: Got it.\nFinal Answer: Done."
    ]
    agent = ReActAgent(llm=MockLLM(responses), tools=TOOLS)
    result = agent.run("Giá HPG")
    assert "Done." in result

def test_tc6_api_failure_escalation(monkeypatch):
    # Simulate API error
    monkeypatch.setattr(tools_module, "SIMULATE_API_ERROR", True)
    
    responses = [
        "Thought: Check price.\nAction: GetPrice(VND)",
        "Thought: Hmm, API error, let me retry.\nAction: GetPrice(VND)",
        "Thought: Still error, third retry.\nAction: GetPrice(VND)",
        "Thought: Should not reach here."
    ]
    agent = ReActAgent(llm=MockLLM(responses), tools=TOOLS)
    result = agent.run("Giá VND")
    assert "bảo trì" in result.lower() or "cập nhật chậm" in result.lower()
