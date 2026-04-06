# 📈 Tài liệu Kỹ thuật: VNStock ReAct Agent Hệ thống Hỏi đáp Chứng khoán thông minh

Hệ thống này sử dụng mô hình ngôn ngữ lớn (LLM) kết hợp với cơ chế **ReAct (Reasoning and Acting)** để truy xuất dữ liệu chứng khoán thời gian thực tại thị trường Việt Nam.

## 1. Kiến trúc Hệ thống (Architecture)

Hệ thống hoạt động dựa trên 3 lớp chính:
1.  **UI Layer (Streamlit):** Tiếp nhận câu hỏi và hiển thị kết quả, biểu đồ.
2.  **Agent (LangChain + Gemma4):** Suy luận xem người dùng đang hỏi gì và cần dùng công cụ (Tool) nào.
3.  **Tool Layer (Data Providers):** Truy xuất dữ liệu từ các thư viện nội địa như `vnstock`.

### Luồng xử lý (Flowchart)
```mermaid
 graph TD
    %% Khởi đầu
    Start(User Query) --> Guardrail{Kiểm tra phạm vi câu hỏi<br/>Intent Check}

    %% Tầng 1: Unified Fallback cho các trường hợp Out-of-Scope
    Guardrail -- "Câu hỏi ngoài phạm vi<br/>của Agent" --> UnifiedFallback[Fallback: Unified Out-of-Scope <br/>- Câu hỏi không liên quan]
    UnifiedFallback --> UI_Msg[Hiển thị thông báo hướng <br/>dẫn về đúng phạm vi CK VN]

    %% Tầng 2: Luồng chính (In-Scope)
    Guardrail -- "-Tra cứu giá cổ phiếu, thông tin liên quan<br>-Biểu đồ của cổ phiếu <br/>-Mã CK VN hợp lệ" --> AgentBrain[Gemma 4: ReAct Thinking]
    
    subgraph ReAct_Core [Thinking & Action loop]
        AgentBrain --> Thought[Thought: Xác định Tool cần gọi]
        Thought --> ActionSelection{Chọn Action}
        
        %% Fallback cho Action Name
        ActionSelection -- "Sai tên Tool" --> ActionErr[Fallback: Action Error Handler <br/>Gửi feedback cho Agent sửa lỗi]
        ActionErr --> AgentBrain
        
        %% Thực thi Tool & Xử lý lỗi API
        ActionSelection -- "Đúng Tool" --> Execute[Gọi API: GetPrice / <br/>CreateChart / GetStockID]
        Execute --> APICheck{Kết quả API?}
        
        APICheck -- "Lỗi (404, 500, Timeout)" --> APIFallback[Fallback: API Error Handler <br/>- Gửi feedback cho Agent <br/>- Chuyển sang Data Source dự phòng]
        APIFallback --> APICheck
    end

    %% Tầng 3: Observation & Output
    APICheck -- "Success" --> Observation[Observation: Nhận dữ liệu]
    Observation --> FinalThought[Thought: Tổng hợp kết quả]
    FinalThought --> FinalAnswer[Gemma 4: Trả lời người dùng]

    %% Tầng 4: Phao cứu sinh cuối (Human Escalation)
    APIFallback -- "Thất bại sau 3 lần" --> HumanEsc[Fallback: Human Escalation <br/>Báo lỗi hệ thống & kết nối người thật]
    
    %% Kết thúc
    FinalAnswer & UI_Msg & HumanEsc --> End(Hiển thị UI)
```

---

## 4. Danh sách Testcases Kiểm thử

| STT | Câu hỏi người dùng | Hành động mong đợi của Agent |
| :--- | :--- | :--- |
| 1 | "Giá FPT hôm nay" | Gọi `GetPrice`, trả về số tiền VND. |
| 2 | "Vẽ biểu đồ cho mã SSI" | Gọi `CreateChart`, trả về biểu đồ nến của cổ phiếu trong vòng 30 ngày. |
| 3 | "Thông tin cổ phiếu FPT" | Gọi `GetStockInfo` để kiểm tra các thông tin của mã cổ phiếu. |
| 4 | "Mua 1000 cổ phiếu VCB" | Nhận diện Out-of-scope (VN Stock) -> Trả lời chung chung hoặc xin lỗi. |
| 5 | "API VNDirect bị bảo trì" | Agent thử lại hoặc báo "Dữ liệu đang cập nhật chậm". |
---

## 5. Hướng dẫn cài đặt nhanh
1. Cài đặt thư viện: `pip install streamlit langchain-google-genai plotly pandas`.
2. Lấy API Key từ [Google AI Studio](https://aistudio.google.com/).
3. Chạy lệnh: `streamlit run app.py`.

---
*Tài liệu này cung cấp nền tảng vững chắc để bạn xây dựng một ứng dụng Fintech thực thụ với khả năng suy luận mạnh mẽ.* Bạn có muốn tôi viết chi tiết phần code tích hợp thư viện `vnstock` để lấy dữ liệu thật từ các sàn HOSE/HNX không?