# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Bùi Lâm Tiến
- **Student ID**: 2A202600004
- **Date**: 06/04/2026

---

## I. Technical Contribution (15 Points)

*Phát triển source code ban đầu của dự án và triển khai tool GetPrice.*

- **Modules Implemented**:
  - `src/agent/tools.py` — Phát triển tool **GetPrice**: truy xuất giá cổ phiếu realtime từ vnstock API, sử dụng `quote.intraday()` với `page_size=10000` để lấy tick giao dịch mới nhất.
  - `src/agent/agent.py` — Xây dựng cấu trúc ban đầu của ReAct Agent loop, bao gồm hệ thống Guardrail (Intent Check) và Error Handling.
  - `app.py` — Thiết kế giao diện Streamlit UI ban đầu để kết nối Agent với người dùng.

- **Code Highlights**:

```python
# src/agent/tools.py — GetPrice function
def GetPrice(symbol: str) -> str:
    symbol = symbol.upper().strip()
    
    # Validate: loại bỏ ký tự không hợp lệ
    if not symbol.isalpha() or len(symbol) < 2 or len(symbol) > 5:
        return f"Mã cổ phiếu '{symbol}' không hợp lệ. Mã hợp lệ gồm 2-5 chữ cái."
    
    try:
        df = Vnstock().stock(symbol=symbol, source='VCI').quote.intraday(
            symbol=symbol, page_size=10000, show_log=False
        )
        if df is None or df.empty:
            return f"Không tìm thấy dữ liệu giá realtime cho mã {symbol}."
        
        last_row = df.iloc[-1]
        latest_price = last_row["price"] * 1000
        # ... format thời gian GMT+7
        return f"Giá hiện tại của {symbol} (cập nhật lúc {formatted_time} GMT+7) là {latest_price:,.0f} VND"
    except ConnectionError:
        raise  # Lỗi mạng thật → cho Agent retry
    except Exception as e:
        # Lỗi do mã không tồn tại → trả kết quả thân thiện, KHÔNG retry
        return f"Không thể tra cứu mã '{symbol}'. Mã này có thể không tồn tại."
```

- **Documentation**: Tool `GetPrice` là hàm được Agent gọi thông qua vòng lặp ReAct khi LLM quyết định cần tra cứu giá. Agent parse được `Action: GetPrice(FPT)` từ output của LLM → hàm `execute_tool_logic()` map tên tool → gọi hàm `GetPrice("FPT")` → kết quả trả về được gắn vào `Observation:` để LLM đưa ra `Final Answer`.

---

## II. Debugging Case Study (10 Points)

*Phân tích lỗi khi người dùng nhập sai mã cổ phiếu (VD: "Giá HPN").*

- **Problem Description**: Khi người dùng hỏi giá của một mã cổ phiếu **không tồn tại** trên sàn VN (ví dụ: `GetPrice(HPN)`), vnstock API ném ra exception. Hàm `GetPrice` ban đầu bọc **tất cả exception** thành `ConnectionError`:
  ```python
  # CODE CŨ (BỊ LỖI)
  except Exception as e:
      raise ConnectionError(f"API VNDirect lỗi: {str(e)}")
  ```
  Điều này khiến Agent hiểu nhầm rằng **API bị lỗi mạng** → kích hoạt **API Error Handler** → retry 3 lần → tất cả đều fail → chuyển sang **Human Escalation** ("Dữ liệu API đang cập nhật chậm..."). Kết quả: người dùng không bao giờ được thông báo rằng **mã cổ phiếu bị sai**, mà chỉ thấy thông báo lỗi API vô nghĩa.

- **Log Source**: Trace panel trên Streamlit UI hiển thị:
  ```
  Step 1:  API_ERROR (retry 1/3) — Action: GetPrice(HPN)
  Step 2:  API_ERROR (retry 2/3) — Action: GetPrice(HPN)  
  Step 3:  HUMAN_ESCALATION — "Dữ liệu API đang cập nhật chậm..."
  ```

- **Diagnosis**: Nguyên nhân gốc rễ là hàm `GetPrice` **không phân biệt** giữa 2 loại lỗi:
  1. **Lỗi mạng/API thật sự** (server down, timeout) → nên retry
  2. **Lỗi do dữ liệu input sai** (mã không tồn tại) → nên trả kết quả ngay, không retry

  Khi `except Exception as e` wrap tất cả thành `ConnectionError`, hệ thống Agent (`agent.py`) hiểu tất cả là lỗi API → `api_failure_count += 1` → sau 3 lần → Human Escalation. Đây là lỗi thiết kế ở **Tool layer**, không phải lỗi do LLM hay prompt.

- **Solution**: Sửa hàm `GetPrice` trong `src/agent/tools.py`:
  1. **Thêm validation đầu vào**: Kiểm tra `symbol.isalpha()` và độ dài 2-5 ký tự trước khi gọi API.
  2. **Tách exception**: Chỉ `raise ConnectionError` cho lỗi mạng thật. Các exception khác (mã không tồn tại, dữ liệu rỗng) trả về message thân thiện dạng `return` → Agent nhận được trong `Observation` và tự ra `Final Answer` thông báo cho user.

  ```python
  # CODE MỚI (ĐÃ FIX)
  except ConnectionError:
      raise  # Lỗi mạng thật → cho Agent retry
  except Exception as e:
      return f"Không thể tra cứu mã '{symbol}'. Mã không tồn tại."
  ```

  **Kết quả sau fix**: Agent nhận Observation thân thiện → trả lời ngay: *"Mã XYZ không tồn tại trên sàn VN"* thay vì retry 3 lần rồi báo lỗi hệ thống.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Khối `Thought` giúp Agent **lập kế hoạch trước khi hành động**. Ví dụ khi user hỏi "Giá FPT", Baseline chatbot sẽ bịa một con số (hallucinate) từ dữ liệu training cũ, trong khi ReAct Agent suy nghĩ: *"Tôi cần dùng GetPrice để lấy giá realtime"* → gọi Tool → trả lời chính xác kèm timestamp. Cơ chế Thought tạo ra **tuyến tính suy luận có kiểm soát**, giúp câu trả lời có trách nhiệm hơn.

2. **Reliability**: Agent hoạt động **kém hơn** Chatbot trong các trường hợp:
   - **Câu hỏi đơn giản không cần tool** (VD: "Chứng khoán là gì?"): Agent tốn thêm 1 bước Intent Check + 1 bước ReAct loop, chậm hơn ~5-10 giây so với Baseline trả lời ngay.
   - **Khi LLM sai format**: Nếu model không viết đúng `Action: GetPrice(FPT)` mà viết `Action: get_price FPT`, parser không parse được → Agent tốn thêm bước → latency cao hơn.
   - **Gemma 4 bị hạn chế output tokens**: Khi prompt dài (nhiều observation tích lũy), output bị cắt ngắn giữa chừng → Agent mất khả năng ra Final Answer.

3. **Observation**: Feedback từ môi trường (Observation) đóng vai trò **bản lề** trong vòng lặp ReAct. Khi Tool trả về dữ liệu thành công, LLM ngay lập tức chuyển sang `Final Answer`. Khi Tool trả về lỗi, Observation cung cấp context để LLM tự sửa (VD: "Sai tên Tool, hãy dùng GetPrice thay vì Get_Price"). Nếu không có Observation, LLM sẽ tự bịa kết quả hoặc lặp vô hạn.

---

## IV. Future Improvements (5 Points)

- **Scalability**: Sử dụng **async/await** cho các tool call để hỗ trợ multiple users đồng thời. Hiện tại mỗi lần gọi vnstock API mất ~2-5 giây blocking, trong môi trường production cần chuyển sang asynchronous queue (Celery hoặc asyncio) để không block Streamlit event loop.

- **Safety**: Triển khai **Supervisor LLM** (một model nhỏ hơn như Gemini Flash) để audit output của Agent trước khi gửi cho user. Supervisor kiểm tra: (1) câu trả lời có chứa thông tin sai lệch về tài chính không, (2) có bị prompt injection không, (3) có tuân thủ format không. Ngoài ra cần thêm rate limiting cho API calls để tránh bị abuse.

- **Performance**: Tích hợp **Vector Database** (Pinecone/ChromaDB) để cache kết quả tra cứu gần đây. Khi user hỏi "Giá FPT" 10 lần trong 1 phút, chỉ cần gọi API 1 lần và cache lại 60 giây. Điều này giảm đáng kể latency và chi phí API. Ngoài ra, có thể dùng **tool retrieval** bằng embedding similarity thay vì hard-code 3 tools, cho phép mở rộng lên hàng trăm tools mà không tốn token để liệt kê tất cả trong system prompt.

---
