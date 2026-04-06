# 📈VNStock ReAct Agent

Hệ thống Hỏi đáp Chứng khoán thông minh sử dụng mô hình ngôn ngữ lớn (LLM) kết hợp với cơ chế **ReAct (Reasoning and Acting)** để truy xuất dữ liệu chứng khoán thời gian thực tại thị trường Việt Nam.

---

## 📁 Cấu trúc Dự án (Repository Structure)

```
day03_E402_group_5b/
├── app.py                          # 🖥️ Streamlit UI — Giao diện người dùng chính
├── requirements.txt                # 📦 Danh sách thư viện cần cài đặt
├── .env                            # 🔑 Biến môi trường (API keys)
├── .env.example                    # 📝 Template biến môi trường
├── .gitignore
│
├── src/                            # 🧠 Source code chính
│   ├── agent/
│   │   ├── agent.py                # 🤖 ReAct Agent — Vòng lặp Thought → Action → Observation
│   │   ├── tools.py                # 🔧 Tool Layer — GetPrice, CreateChart, GetStockInfo
│   │   └── gemini_demo.py          # 🧪 Demo script cho Gemini
│   │
│   ├── core/
│   │   ├── llm_provider.py         # 📐 Abstract Base Class cho LLM Providers
│   │   ├── gemini_provider.py      # 🔗 Google Gemini / Gemini-2.5-Flash API Provider
│   │   ├── openai_provider.py      # 🔗 OpenAI Provider (mở rộng)
│   │   └── local_provider.py       # 🔗 Local GGUF Provider (mở rộng)
│   │
│   └── telemetry/
│       ├── logger.py               # 📋 Event Logger — Ghi log sự kiện Agent
│       └── metrics.py              # 📊 Metrics — Thu thập dữ liệu hiệu suất
│
├── tests/
│   ├── test_agent_loop.py          # ✅ Unit tests cho ReAct loop
│   ├── test_gemma.py               # ✅ Tests cho Gemma model
│   └── test_local.py               # ✅ Tests cho Local provider
│
├── logs/                           # 📂 Thư mục lưu log runtime
├── report/                         # 📂 Thư mục báo cáo
├── EVALUATION.md                   # 📋 Tiêu chí đánh giá
├── SCORING.md                      # 📋 Bảng chấm điểm
└── INSTRUCTOR_GUIDE.md             # 📋 Hướng dẫn giảng viên
```

---

## 1. Kiến trúc Hệ thống (Architecture)

Hệ thống hoạt động dựa trên 3 lớp chính:
1.  **UI Layer (Streamlit):** Tiếp nhận câu hỏi và hiển thị kết quả, biểu đồ, trace.
2.  **Agent (ReAct + Gemma 4):** Suy luận xem người dùng đang hỏi gì và cần dùng công cụ (Tool) nào.
3.  **Tool Layer (vnstock):** Truy xuất dữ liệu realtime từ VCI/VNDirect.

### Các thành phần chính

| Thành phần | File | Mô tả |
|---|---|---|
| **Streamlit UI** | `app.py` | Giao diện chat, sidebar cấu hình, hiển thị trace/cost/security |
| **ReAct Agent** | `src/agent/agent.py` | Vòng lặp ReAct, Guardrail, Intent Check, Error Handling |
| **Tools** | `src/agent/tools.py` | `GetPrice` (giá realtime), `CreateChart` (biểu đồ nến), `GetStockInfo` (thông tin công ty) |
| **LLM Provider** | `src/core/gemini_provider.py` | Kết nối Gemini API, hỗ trợ Gemma 4 và Gemini 2.5 Flash |
| **Telemetry** | `src/telemetry/logger.py` | Ghi log sự kiện cho debugging và monitoring |

### Luồng xử lý (Flowchart)
```mermaid
 graph TD
    %% Khởi đầu
    Start(User Query) --> Guardrail{Kiểm tra phạm vi câu hỏi<br/>Intent Check}

    %% Tầng 1: Unified Fallback cho các trường hợp Out-of-Scope
    Guardrail -- "Câu hỏi ngoài phạm vi<br/>của Agent" --> UnifiedFallback[Fallback: Unified Out-of-Scope <br>- Câu hỏi không liên quan]
    UnifiedFallback --> UI_Msg[Hiển thị thông báo hướng <br>dẫn về đúng phạm vi CK VN]

    %% Tầng 2: Luồng chính (In-Scope)
    Guardrail -- "-Tra cứu giá cổ phiếu, thông tin liên quan<br>-Biểu đồ của cổ phiếu <br/>-Mã CK VN hợp lệ" --> AgentBrain[Gemma 4: ReAct Thinking]
    
    subgraph ReAct_Core [Thinking & Action loop]
        AgentBrain --> Thought[Thought: Xác định Tool cần gọi]
        Thought --> ActionSelection{Chọn Action}
        
        %% Fallback cho Action Name
        ActionSelection -- "Sai tên Tool" --> ActionErr[Fallback: Action Error Handler <br>Gửi feedback cho Agent sửa lỗi]
        ActionErr --> AgentBrain
        
        %% Thực thi Tool & Xử lý lỗi API
        ActionSelection -- "Đúng Tool" --> Execute[Gọi API: GetPrice / <br>CreateChart / GetStockInfo]
        Execute --> APICheck{Kết quả API?}
        
        APICheck -- "Lỗi (404, 500, Timeout)" --> APIFallback[Fallback: API Error Handler <br>- Gửi feedback cho Agent <br>- Chuyển sang Data Source dự phòng]
        APIFallback --> APICheck
    end

    %% Tầng 3: Observation & Output
    APICheck -- "Success" --> Observation[Observation: Nhận dữ liệu]
    Observation --> FinalThought[Thought: Tổng hợp kết quả]
    FinalThought --> FinalAnswer[Gemma 4: Trả lời người dùng]

    %% Tầng 4: Phao cứu sinh cuối (Human Escalation)
    APIFallback -- "Thất bại sau 3 lần" --> HumanEsc[Fallback: Human Escalation <br>Báo lỗi hệ thống & kết nối người thật]
    
    %% Kết thúc
    FinalAnswer & UI_Msg & HumanEsc --> End(Hiển thị UI)
```

---

## 2. Tính năng nổi bật

### 🤖 ReAct Agent
- Vòng lặp **Thought → Action → Observation** tự động
- Tối đa 5 bước suy luận trước khi trả lời
- Guardrail 2 lớp: **Blacklist keyword** + **LLM Whitelist** (chỉ cho phép 3 tác vụ)

### 🔧 3 Tools tích hợp vnstock
| Tool | Chức năng | API |
|---|---|---|
| `GetPrice(symbol)` | Giá realtime (intraday) | `quote.intraday()` |
| `CreateChart(symbol)` | Biểu đồ nến theo giờ | `Quote.history()` + Plotly |
| `GetStockInfo(symbol)` | Thông tin tổng quan công ty | `Company.overview()` |

### 📊 Trace & Evaluation Panel
- **ReAct Trace**: Hiển thị từng bước Thought/Action/Observation trên UI
- **Cost Dashboard**: Token usage (prompt/completion), ước tính chi phí USD
- **Security Audit**: Phát hiện Prompt Injection, blocked keywords
- **Evaluation Metrics**: Steps, Latency, LLM Calls

### 🧠 Chế độ A/B Testing
- **ReAct Agent**: Sử dụng Tool, dữ liệu chính xác realtime
- **Baseline**: LLM trả lời trực tiếp không Tool (để so sánh hallucination)

### 🛡️ Error Handling
- **Action Error Handler**: Tự sửa khi gõ sai tên Tool
- **API Error Handler**: Retry tối đa 3 lần
- **Human Escalation**: Chuyển sang thông báo lỗi hệ thống sau 3 lần thất bại

---

## 3. Công nghệ sử dụng

| Công nghệ | Phiên bản | Vai trò |
|---|---|---|
| Python | 3.10+ | Ngôn ngữ chính |
| Streamlit | latest | Giao diện web |
| Google Generative AI | ≥0.5.0 | Kết nối Gemini / Gemma 4 API |
| vnstock | latest | Dữ liệu chứng khoán VN realtime |
| Plotly | latest | Biểu đồ nến kỹ thuật |
| Pandas | latest | Xử lý dữ liệu |
| python-dotenv | ≥1.0.0 | Quản lý biến môi trường |
| pytest | ≥7.4.0 | Unit testing |

---

## 4. Danh sách Testcases Kiểm thử

| STT | Câu hỏi người dùng | Hành động mong đợi của Agent |
| :--- | :--- | :--- |
| 1 | "Giá FPT hôm nay" | Gọi `GetPrice`, trả về số tiền VND kèm timestamp. |
| 2 | "Vẽ biểu đồ cho mã SSI" | Gọi `CreateChart`, trả về biểu đồ nến theo giờ. |
| 3 | "Thông tin cổ phiếu FPT" | Gọi `GetStockInfo` để kiểm tra các thông tin của mã cổ phiếu. |
| 4 | "Mua 1000 cổ phiếu VCB" | Nhận diện Out-of-scope → Trả lời xin lỗi. |
| 5 | "API VNDirect bị bảo trì" | Agent thử lại 3 lần hoặc báo "Dữ liệu đang cập nhật chậm". |

---

## 5. Hướng dẫn cài đặt & chạy

### Yêu cầu
- Python 3.10+
- API Key từ [Google AI Studio](https://aistudio.google.com/)
- API Key từ [vnstock](https://vnstock.site/) (gói miễn phí)

### Cài đặt
```bash
# Clone repo
git clone <repo-url>
cd day03_E402_group_5b

# Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài thư viện
pip install -r requirements.txt
```

### Cấu hình API Keys
```bash
# Copy file mẫu
cp .env.example .env

# Sửa file .env và điền API keys:
# GEMINI_API_KEY=your_gemini_api_key_here
# VNSTOCK_API_KEY=your_vnstock_api_key_here
```

### Đăng ký vnstock (chỉ chạy 1 lần)
```python
from vnstock import register_user
register_user('your_vnstock_api_key')
```

### Chạy ứng dụng
```bash
streamlit run app.py
```

### Chạy tests
```bash
pytest tests/ -v
```

---

## 6. Biến môi trường

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | API key từ Google AI Studio |
| `VNSTOCK_API_KEY` | ✅ | API key từ vnstock.site |
| `DEFAULT_MODEL` | ❌ | Model mặc định (default: `gemma-4-31b-it`) |
| `LOG_LEVEL` | ❌ | Mức log (default: `INFO`) |

---

*Dự án được phát triển trong khuôn khổ AI VinUni - Day 3. Sử dụng kiến trúc ReAct Agent để xây dựng một chatbot có khả năng truy xuất dữ liệu chứng khoán.