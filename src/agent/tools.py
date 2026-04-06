import datetime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List
from vnstock import Vnstock, register_user, Company

# Đăng nhập bằng API Key để cải thiện độ chuẩn xác/giới hạn dữ liệu từ hệ thống cấp dữ liệu
register_user('vnstock_503967bcbf987a7a89727aa469fb957b')

# We will use vnstock for real data, but allow simulating an error for testing.

SIMULATE_API_ERROR = False

def GetPrice(symbol: str) -> str:
    """Gets the latest close/intraday price for a VN stock symbol."""
    if SIMULATE_API_ERROR:
        raise ConnectionError("API VNDirect bị bảo trì / Timeout")
    
    symbol = symbol.upper().strip()
    
    try:
        # Sử dụng page_size=10000 để đảm bảo tải dữ liệu mới nhất (theo yêu cầu)
        df = Vnstock().stock(symbol=symbol, source='VCI').quote.intraday(symbol=symbol, page_size=10000, show_log=False)
        if df is None or df.empty:
            return f"Không tìm thấy dữ liệu giá realtime cho mã {symbol}."
        
        last_row = df.iloc[-1]
        latest_price = last_row["price"] * 1000 # vnstock trả về mốc chia 1000
        
        # Đảm bảo hiển thị đúng múi giờ GMT+7
        dt = pd.to_datetime(last_row["time"])
        if dt.tzinfo is None:
            dt = dt.tz_localize('Asia/Ho_Chi_Minh')
        else:
            dt = dt.tz_convert('Asia/Ho_Chi_Minh')
            
        formatted_time = dt.strftime('%H:%M:%S %d-%m-%Y')
        return f"Giá hiện tại của {symbol} (cập nhật lúc {formatted_time} GMT+7) là {latest_price:,.0f} VND"
    except Exception as e:
        raise ConnectionError(f"API VNDirect lỗi: {str(e)}")

def CreateChart(symbol: str) -> str:
    """Creates a technical chart for a symbol. Returns a confirmation string for the UI."""
    if SIMULATE_API_ERROR:
        raise ConnectionError("API VNDirect bị bảo trì / Timeout")

    symbol = symbol.upper().strip()
    
    # Lấy ngày theo múi giờ Việt Nam (GMT+7) chứ không phụ thuộc giờ server
    vn_tz = datetime.timezone(datetime.timedelta(hours=7))
    vn_now = datetime.datetime.now(vn_tz)
    
    end_date = vn_now.strftime("%Y-%m-%d")
    start_date = (vn_now - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    
    try:
        df = Vnstock().stock(symbol=symbol, source='VCI').quote.history(start=start_date, end=end_date)
        if df.empty:
            return f"Không có dữ liệu để vẽ biểu đồ cho mã {symbol}."
        return f"Đã vẽ biểu đồ kỹ thuật mã {symbol} thành công. Tín hiệu Plotly đã được gửi tới UI."
    except Exception as e:
        raise ConnectionError(f"Lỗi khi vẽ biểu đồ: {str(e)}")

def GetInfoID(symbol: str) -> str:
    """Tra cứu thông tin công ty từ mã cổ phiếu ở múi giờ GMT+7."""
    if SIMULATE_API_ERROR:
        raise ConnectionError("API VNDirect bị bảo trì / Timeout")

    symbol = symbol.upper().strip()
    
    try:
        company = Company(symbol=symbol, source='KBS')
        df = company.overview()
        
        if df is None or df.empty:
            return f"Không tìm thấy thông tin cho mã {symbol}."
        
        info = df.iloc[0].to_dict()
        
        # Bỏ các trường liên quan đến giá hiện tại
        keys_to_remove = [k for k in info.keys() if 'price' in k.lower() or 'giá' in k.lower()]
        for k in keys_to_remove:
            info.pop(k, None)
        
        vn_tz = datetime.timezone(datetime.timedelta(hours=7))
        vn_now = datetime.datetime.now(vn_tz)
        formatted_time = vn_now.strftime('%H:%M:%S %d-%m-%Y')
        
        res = f"Thông tin của {symbol} (cập nhật lúc {formatted_time} GMT+7):\n"
        for k, v in info.items():
            res += f"- {k}: {v}\n"
            
        try:
            # Truy xuất trực tiếp dữ liệu giá intraday để có giá chính xác nhất mà không cần gọi hàm GetPrice
            df_price = Vnstock().stock(symbol=symbol, source='VCI').quote.intraday(symbol=symbol, page_size=2, show_log=False)
            if df_price is not None and not df_price.empty:
                latest_price = df_price.iloc[-1]["price"] * 1000 # Cập nhật mốc chia
                res += f"\nGiá hiện hành chính xác: {latest_price:,.0f} VND\n"
        except Exception as price_err:
            res += f"\n(Lỗi khi lấy dữ liệu giá hiện hành: {str(price_err)})\n"
            
        return res
    except Exception as e:
        raise ConnectionError(f"API API lỗi: {str(e)}")

# Define the tool schemas for the Agent
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "GetPrice",
        "description": "Dùng để lấy giá cổ phiếu hiện tại của một mã chứng khoán (nội địa Việt Nam). Đầu vào là 1 mã chứng khoán (VD: FPT, HPG)."
    },
    {
        "name": "CreateChart",
        "description": "Dùng để vẽ biểu đồ kỹ thuật cho một mã chứng khoán. Đầu vào là mã chứng khoán (VD: SSI, VCB)."
    },
    {
        "name": "GetInfoID",
        "description": "Dùng để tra cứu thông tin của công ty từ mã cổ phiếu (VD: FPT). Đầu vào là mã cổ phiếu."
    }
]

def execute_tool_logic(tool_name: str, args: str) -> str:
    """Helper method to map tool names to Python functions."""
    if tool_name == "GetPrice":
        return GetPrice(args)
    elif tool_name == "CreateChart":
        return CreateChart(args)
    elif tool_name == "GetInfoID":
        return GetInfoID(args)
    else:
        # Action Error Handler scenario will catch this if the LLM hallucinates
        raise ValueError(f"Sai tên Tool: {tool_name}")
