import datetime
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from vnstock import Vnstock, register_user, Quote
import os

# In a real environment, vnstock fetches from VNDirect/TCBS/SSI.
# We will use vnstock for real data, but allow simulating an error for testing.

SIMULATE_API_ERROR = False

# Đăng ký tài khoản dùng thử (chỉ cần chạy 1 lần)
register_user(os.getenv("VNSTOCK_API_KEY"))

def GetPrice(symbol: str) -> str:
    """Gets the latest close price for a VN stock symbol."""
    if SIMULATE_API_ERROR:
        raise ConnectionError("API VNDirect bị bảo trì / Timeout")
    
    symbol = symbol.upper().strip()
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        quote = Quote(symbol=symbol, source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='d')
        if df.empty:
            return f"Không tìm thấy dữ liệu giá cho mã {symbol}."
        latest_price = df.iloc[-1]["close"] * 1000 # vnstock trả về giá x1000
        return f"Giá hiện tại của {symbol} là {latest_price:,.0f} VND"
    except Exception as e:
        raise ConnectionError(f"API VNDirect lỗi: {str(e)}")

def CreateChart(symbol: str) -> str:
    """Creates a technical chart for a symbol. Returns a confirmation string for the UI."""
    if SIMULATE_API_ERROR:
        raise ConnectionError("API VNDirect bị bảo trì / Timeout")

    symbol = symbol.upper().strip()
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
    
    try:
        quote = Quote(symbol=symbol, source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='1H')
        if df.empty:
            return f"Không có dữ liệu để vẽ biểu đồ cho mã {symbol}."
            
        fig = go.Figure(data=[go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        
        fig.update_layout(title=f'Biểu đồ nến theo giờ {symbol}', xaxis_title='Thời gian', yaxis_title='Giá', template='plotly_dark', xaxis_rangeslider_visible=False)
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]), # Ẩn thứ 7, Chủ Nhật
            dict(bounds=[15, 9], pattern="hour"), # Ẩn từ 15h chiều đến 9h sáng hôm sau
            dict(bounds=[11.5, 13], pattern="hour") # Ẩn giờ nghỉ trưa 11h30 - 13h00
        ])
        
        if "temp_charts" not in st.session_state:
            st.session_state.temp_charts = []
        st.session_state.temp_charts.append(fig)
        
        return f"Đã vẽ biểu đồ kỹ thuật mã {symbol} thành công. Tín hiệu Plotly đã được gửi tới UI."
    except Exception as e:
        raise ConnectionError(f"Lỗi khi vẽ biểu đồ: {str(e)}")

def GetStockID(company_name: str) -> str:
    """Tra cứu mã cổ phiếu từ tên công ty."""
    if SIMULATE_API_ERROR:
        raise ConnectionError("API VNDirect bị bảo trì / Timeout")

    # A mock dictionary for demonstration, or we can use vnstock if it has a fuzzy search.
    # vnstock's listing feature retrieves all stocks, but we'll mock a few for speed.
    companies = {
        "fpt": "FPT",
        "hòa phát": "HPG",
        "hoa sen": "HSG",
        "vietcombank": "VCB",
        "ssi": "SSI"
    }
    
    for key in companies:
        if key in company_name.lower():
            return f"Mã cổ phiếu của {company_name} là {companies[key]}"
            
    return f"Không tìm thấy mã cổ phiếu hợp lệ cho công ty: {company_name}."

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
        "name": "GetStockID",
        "description": "Dùng để tra cứu mã cổ phiếu khi người dùng chỉ cung cấp tên công ty. Đầu vào là tên công ty."
    }
]

def execute_tool_logic(tool_name: str, args: str) -> str:
    """Helper method to map tool names to Python functions."""
    if tool_name == "GetPrice":
        return GetPrice(args)
    elif tool_name == "CreateChart":
        return CreateChart(args)
    elif tool_name == "GetStockID":
        return GetStockID(args)
    else:
        # Action Error Handler scenario will catch this if the LLM hallucinates
        raise ValueError(f"Sai tên Tool: {tool_name}")
