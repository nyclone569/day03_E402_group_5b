# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: Stock Agent (E402_5B)
- **Team Members**: Võ Thiên Phú (2A202600336), Bùi Lâm Tiến (2A202600004), Trương Đăng Nghĩa (2A202600437)
- **Deployment Date**: 2026-04-06

---

## 1. Executive Summary

Chúng tôi xây dựng một **VNStock ReAct Agent** — hệ thống hỏi đáp chứng khoán thông minh cho thị trường Việt Nam, sử dụng mô hình ngôn ngữ lớn kết hợp cơ chế **ReAct (Reasoning and Acting)** để tra cứu giá cổ phiếu, vẽ biểu đồ kỹ thuật và xử lý các câu hỏi phức hợp nhiều bước. So với baseline Chatbot (chỉ trả lời dựa trên knowledge cố định), Agent có thể truy vấn dữ liệu thời gian thực và giải quyết các tác vụ đa bước.

- **Success Rate**: 80% trên 5 test case tiêu chuẩn (4/5 câu hỏi được xử lý đúng)
- **Key Outcome**: Agent giải quyết được **100% test case multi-step** (so sánh 2 mã cổ phiếu, vẽ biểu đồ kết hợp) — đây là loại câu hỏi mà Chatbot baseline hoàn toàn thất bại do phải gọi tool nhiều lần và tổng hợp kết quả.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

Hệ thống hoạt động theo vòng lặp **Thought → Action → Observation** được điều phối bởi `ReActAgent` trong `src/agent/agent.py`.

```
User Query
    │
    ▼
[Guardrail] ──(Out-of-scope)──► Fallback Response
    │
    ▼ (In-scope: VN Stock queries)
[Gemma 4 LLM - Thought]
    │
    ▼
[Action Parsing]
    ├──(Invalid tool name)──► Action Error Handler ──► Retry
    └──(Valid tool)──────────► Tool Execution
                                    │
                                    ▼
                              [Observation]
                                    │
                                    ▼
                              [Next Thought] ──(Final Answer)──► UI
```

- **`max_steps = 5`**: Giới hạn vòng lặp để tránh vô hạn và kiểm soát chi phí token.
- **Telemetry**: Mỗi sự kiện `AGENT_START` / `AGENT_END` được ghi log qua `src/telemetry/logger.py` và `src/telemetry/metrics.py`.
- **LLM Providers**: Hỗ trợ đa provider (Gemini, OpenAI, Local) qua abstraction `src/core/llm_provider.py`.

### 2.2 Tool Definitions (Inventory)

| Tool Name | Input Format | Use Case |
| :--- | :--- | :--- |
| `GetPrice` | `string` — mã 3 chữ cái (VD: `HPG`) | Lấy giá hiện tại của cổ phiếu VN (đơn vị VND) |
| `CreateChart` | `string` — mã 3 chữ cái (VD: `SSI`) | Vẽ biểu đồ nến Nhật (Candlestick) qua Plotly, kích hoạt trên Streamlit UI |
| `GetInfo` | `string` — mã 3 chữ cái (VD: `VNM`) | Lấy thông tin tổng quan công ty: tên đầy đủ, ngành, vốn hoá, sàn niêm yết — sử dụng **Gemma 4** để tóm tắt và giải thích |

### 2.3 LLM Providers Used

- **Primary (ReAct Loop + GetInfo)**: Google **Gemma 4** (`gemma-4` via `langchain-google-genai`, `temperature=0`) — mô hình chính cho toàn bộ vòng lặp Thought-Action-Observation và công cụ `GetInfo`
- **Secondary (Backup)**: OpenAI GPT-4o (qua `src/core/openai_provider.py`) — dùng khi Gemma 4 quota bị giới hạn

> **Lý do chọn Gemma 4**: Gemma 4 có khả năng lý luận mạnh mẽ hơn Gemini Flash trong các tác vụ tóm tắt thông tin tài chính cấu trúc (`GetInfo`), đồng thời vẫn giữ được tốc độ phản hồi chấp nhận được cho ứng dụng thời gian thực.

---

## 3. Telemetry & Performance Dashboard

*Dữ liệu đo đạc được thu thập qua module `src/telemetry/` trong lần chạy final test (5 test cases, 3 lần mỗi case).*

| Metric | Value |
| :--- | :--- |
| **Average Latency (P50)** | ~1,500 ms |
| **Max Latency (P99)** | ~5,200 ms (test case multi-step `GetInfo` + `GetPrice`, 3 vòng lặp) |
| **Average Tokens per Task** | ~480 tokens (prompt + completion, tăng do output `GetInfo` phong phú hơn) |
| **Average ReAct Steps per Task** | 2.1 steps |
| **Total Cost of Test Suite (18 runs — 6 test cases × 3)** | ~$0.04 (Gemma 4 pricing) |
| **Tool Call Success Rate** | 94.4% (17/18 calls) |

> **Nhận xét:** Latency tăng đáng kể ở các câu hỏi cần 2+ vòng lặp do phải chờ API trả về 2 lần. Token count ổn định nhờ system prompt được tối ưu ngắn gọn.

---

## 4. Root Cause Analysis (RCA) - Failure Traces

*Phân tích sâu các trường hợp Agent thất bại.*

### Case Study 1: Hallucinated Tool Argument — Mã cổ phiếu không hợp lệ

- **Input**: `"Giá cổ phiếu của Hòa Phát hôm nay?"`
- **Observation**: Agent gọi `GetPrice(symbol="HoaPhat")` thay vì `GetPrice(symbol="HPG")`, dẫn đến lỗi `"không tìm thấy"`.
- **Root Cause**: System prompt thiếu ví dụ Few-Shot ánh xạ tên công ty → mã ticker. LLM tự sinh tên công ty thay vì mã 3 chữ cái theo quy định của HOSE/HNX.
- **Fix Applied**: Bổ sung 3 ví dụ Few-Shot vào `get_system_prompt()`: `Hòa Phát → HPG`, `Vinamilk → VNM`, `FPT → FPT`.

### Case Study 2: Câu hỏi Out-of-Scope — Agent không nhận diện đúng

- **Input**: `"Dự báo giá vàng thế giới tuần tới?"`
- **Observation**: Agent cố gắng gọi `GetPrice(symbol="XAUUSD")` thay vì trả về thông báo out-of-scope.
- **Root Cause**: Guardrail chưa được cài đặt. Agent diễn giải câu hỏi như một yêu cầu tra cứu chứng khoán.
- **Fix Applied**: Thêm bước kiểm tra intent ở đầu vòng lặp: nếu symbol không có trong danh sách mã VN hợp lệ, trả về fallback message.

---

## 5. Ablation Studies & Experiments

### Experiment 1: System Prompt v1 vs Prompt v2

- **Diff**: Prompt v2 thêm dòng `"Luôn kiểm tra tên tool và format argument trước khi gọi. Mã cổ phiếu VN luôn là 3 chữ cái viết hoa."` + 3 ví dụ Few-Shot.
- **Result**: Giảm lỗi gọi sai tên tool từ 3 lần → 1 lần trong 15 lần chạy (**giảm ~67% tool call errors**). Accuracy tổng thể tăng từ 60% → 80%.

### Experiment 2 (Bonus): Chatbot vs ReAct Agent

| Test Case | Chatbot Result | Agent Result | Winner |
| :--- | :--- | :--- | :--- |
| TC1: Giá FPT hôm nay | ❌ Hallucinated (giá cũ từ training data) | ✅ Correct (gọi `GetPrice`) | **Agent** |
| TC2: So sánh HPG và HSG | ❌ Hallucinated cả 2 giá | ✅ Correct (2 tool calls, tính toán đúng) | **Agent** |
| TC3: Vẽ biểu đồ SSI | ❌ Không thể tạo biểu đồ | ✅ Correct (gọi `CreateChart`) | **Agent** |
| TC4: Dự báo giá vàng | ✅ Trả lời chung "Tôi không biết" | ⚠️ Cố gọi tool (v1) / ✅ Fallback (v2) | Draw (sau fix) |
| TC5: Câu hỏi đơn giản về thị trường | ✅ Correct | ✅ Correct | Draw |
| TC6: "Vinamilk là công ty gì?" | ⚠️ Trả lời chung từ training data (không có ngành, vốn hoá) | ✅ Correct (gọi `GetInfo(VNM)` → Gemma 4 tóm tắt chuẩn xác) | **Agent** |

> **Kết luận**: Agent vượt trội ở các tác vụ cần dữ liệu thời gian thực và multi-step. Chatbot chỉ trả lời đúng những câu hỏi chung chung không cần tool.

---

## 6. Production Readiness Review

*Các yếu tố cần xem xét khi đưa hệ thống này vào môi trường thực tế.*

- **Security**: Sanitize input tool arguments — không cho phép mã cổ phiếu dài quá 10 ký tự, block ký tự đặc biệt để tránh injection.
- **Guardrails**: 
  - Giới hạn `max_steps = 5` để tránh vòng lặp vô hạn và kiểm soát chi phí.
  - Thêm timeout 10s cho mỗi tool call.
  - Human Escalation: Sau 3 lần API thất bại → gửi alert tới admin qua Telegram Bot.
- **Scaling**: 
  - Chuyển sang **LangGraph** để hỗ trợ branching phức tạp hơn (VD: song song gọi nhiều tool).
  - Cache kết quả `GetPrice` và `GetInfo` trong Redis với TTL 60 giây, tránh gọi API trùng lặp.
  - Sử dụng **async tool execution** cho tác vụ đa agent.
  - Tách `GetInfo` thành microservice riêng để scale độc lập (do Gemma 4 inference nặng hơn Flash).
- **Observability**: Mở rộng telemetry hiện tại sang Prometheus + Grafana dashboard để theo dõi latency và error rate theo thời gian thực.

---

> [!NOTE]
> Submit this report by renaming it to `GROUP_REPORT_[TEAM_NAME].md` and placing it in this folder.
