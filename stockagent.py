import streamlit as st
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from newsapi import NewsApiClient

# Load environment variables
load_dotenv()

# Configure API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class FinancialAgent:
    """Financial agent that provides stock information, analysis, and news."""
    
    def __init__(self):
        """Initialize the financial agent with necessary API clients."""
        self.newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None
        # Common Indian stock symbol mappings
        self.indian_stock_mappings = {
            'TATAMOTORS': 'TATAMOTORS.NS',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'MAHINDRA': 'M&M.NS',
            'TATA': 'TATASTEEL.NS',  # Use specific Tata company symbol
            'BAJAJ': 'BAJFINANCE.NS',  # Use specific Bajaj company symbol
        }
    
    def get_valid_symbol(self, ticker: str, exchange: str = "US") -> str:
        """Get valid symbol based on exchange."""
        ticker = ticker.upper().replace(" ", "")
        
        if exchange == "NSE (India)":
            # Check if it's in our mapping first
            if ticker in self.indian_stock_mappings:
                return self.indian_stock_mappings[ticker]
            # If not in mapping, add .NS suffix if not already present
            if not ticker.endswith('.NS'):
                return f"{ticker}.NS"
        elif exchange == "BSE (India)":
            # Remove any existing suffix and add .BO
            base_ticker = ticker.split('.')[0]
            return f"{base_ticker}.BO"
        
        return ticker
    
    def get_stock_price(self, ticker: str, exchange: str = "US") -> Dict[str, Any]:
        """Get current and historical stock prices."""
        try:
            valid_ticker = self.get_valid_symbol(ticker, exchange)
            stock = yf.Ticker(valid_ticker)
            info = stock.info
            
            if not info:
                suggestion = ""
                if exchange in ["NSE (India)", "BSE (India)"]:
                    suggestion = (
                        "For Indian stocks, try:\n"
                        "1. Using the exact symbol (e.g., 'TATAMOTORS' instead of 'TATA')\n"
                        "2. Checking if the stock is listed on the selected exchange\n"
                        "3. Using the complete company name without spaces"
                    )
                raise Exception(f"No data found for ticker {valid_ticker}. {suggestion}")

            # Get current price data
            current_data = {
                "symbol": valid_ticker,
                "name": info.get("shortName", "N/A"),
                "price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "change": info.get("regularMarketChangePercent", "N/A"),
                "previous_close": info.get("regularMarketPreviousClose", "N/A"),
                "open": info.get("regularMarketOpen", "N/A"),
                "day_high": info.get("dayHigh", "N/A"),
                "day_low": info.get("dayLow", "N/A"),
                "volume": info.get("volume", "N/A"),
            }
            
            # Get historical data for the past month
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            hist = stock.history(start=start_date, end=end_date)
            
            return {
                "current": current_data,
                "historical": hist.reset_index().to_dict('records') if not hist.empty else [],
                "hist_df": hist
            }
        except Exception as e:
            return {"error": f"Error fetching stock price: {str(e)}"}
    
    def get_analyst_recommendations(self, ticker: str) -> Dict[str, Any]:
        """Get analyst recommendations for a stock."""
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                return {
                    "recommendations": recommendations.reset_index().to_dict('records'),
                    "recommendations_df": recommendations
                }
            else:
                return {"message": "No analyst recommendations found"}
        except Exception as e:
            return {"error": f"Error fetching analyst recommendations: {str(e)}"}
    
    def get_stock_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get fundamental data for a stock."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get key financial metrics
            fundamentals = {
                "market_cap": info.get("marketCap", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "trailing_pe": info.get("trailingPE", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "beta": info.get("beta", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "50_day_average": info.get("fiftyDayAverage", "N/A"),
                "200_day_average": info.get("twoHundredDayAverage", "N/A"),
                "profit_margins": info.get("profitMargins", "N/A"),
                "revenue_growth": info.get("revenueGrowth", "N/A"),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                "return_on_equity": info.get("returnOnEquity", "N/A"),
                "free_cash_flow": info.get("freeCashflow", "N/A"),
                "operating_cash_flow": info.get("operatingCashflow", "N/A"),
            }
            
            # Get quarterly financials if available
            try:
                quarterly_financials = stock.quarterly_financials
                if quarterly_financials is not None and not quarterly_financials.empty:
                    fundamentals["quarterly_financials"] = quarterly_financials.reset_index().to_dict('records')
                    fundamentals["quarterly_financials_df"] = quarterly_financials
            except:
                pass
                
            return fundamentals
        except Exception as e:
            return {"error": f"Error fetching stock fundamentals: {str(e)}"}
    
    def get_company_news(self, company_name: str, days: int = 7) -> Dict[str, Any]:
        """Get recent news about a company."""
        try:
            # Try using News API if key is available
            if self.newsapi:
                news = self.newsapi.get_everything(
                    q=company_name,
                    language='en',
                    sort_by='publishedAt',
                    from_param=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    to=datetime.now().strftime('%Y-%m-%d'),
                    page_size=10
                )
                
                if news and news.get('articles'):
                    return {"articles": news['articles']}
            
            # Fallback to Yahoo Finance news
            ticker = yf.Ticker(company_name)
            news = ticker.news
            
            if news:
                return {"articles": news}
            else:
                return {"message": "No recent news found"}
        except Exception as e:
            return {"error": f"Error fetching company news: {str(e)}"}

def format_large_number(num):
    """Format large numbers into K, M, B, T format."""
    if not isinstance(num, (int, float)) or pd.isna(num):
        return "N/A"
    
    if abs(num) >= 1_000_000_000_000:  # Trillion
        return f"${num / 1_000_000_000_000:.2f}T"
    elif abs(num) >= 1_000_000_000:  # Billion
        return f"${num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:  # Million
        return f"${num / 1_000_000:.2f}M"
    elif abs(num) >= 1_000:  # Thousand
        return f"${num / 1_000:.2f}K"
    else:
        return f"${num:,.2f}"

def format_percentage(value):
    """Format value as percentage."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"

def main():
    # Set page config
    st.set_page_config(
        page_title="Financial Analysis Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # App title and description
    st.title("📊 Stock Analysis Dashboard")
    st.markdown("""
    Enter a stock ticker symbol to get comprehensive analysis including price data, 
    analyst recommendations, fundamentals, and recent news.
    """)
    
    # Initialize FinancialAgent
    agent = FinancialAgent()
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Stock Selection")
        exchange = st.selectbox(
            "Select Exchange",
            ["US", "NSE (India)", "BSE (India)"],
            index=0
        )
        
        # Add helper text based on exchange
        if exchange in ["NSE (India)", "BSE (India)"]:
            st.markdown("""
            **Indian Stock Examples:**
            - TATAMOTORS (Tata Motors)
            - RELIANCE (Reliance Industries)
            - HDFCBANK (HDFC Bank)
            - INFY (Infosys)
            """)
            
        ticker = st.text_input("Enter Stock Ticker Symbol:", value="AAPL").upper()
        analyze_button = st.button("Analyze Stock", type="primary")
        
        # Show the actual symbol being used
        if exchange in ["NSE (India)", "BSE (India)"]:
            valid_symbol = agent.get_valid_symbol(ticker, exchange)
            st.caption(f"Searching for: {valid_symbol}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides comprehensive stock analysis using data from:
        - Yahoo Finance for stock data (US, NSE, and BSE markets)
        - News API for company news (if API key is provided)
        
        For Indian stocks:
        - Use company name without any suffix
        - Select appropriate exchange (NSE or BSE)
        Example: For Tata Motors, enter 'TATAMOTORS'
        """)
        
    # Store ticker in session state if not already present
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = ""
    
    # Only run analysis when button is clicked or when ticker changes
    if analyze_button or (ticker and ticker != st.session_state.last_ticker):
        st.session_state.last_ticker = ticker
        
        # Show loading spinner
        with st.spinner(f"Analyzing {ticker}..."):
            # Get stock data
            price_data = agent.get_stock_price(ticker, exchange)
            recommendations_data = agent.get_analyst_recommendations(ticker)
            fundamentals_data = agent.get_stock_fundamentals(ticker)
            
            # Get company name for news search
            company_name = price_data.get("current", {}).get("name", ticker)
            news_data = agent.get_company_news(company_name)
        
        # Check if there was an error
        if "error" in price_data:
            st.error(price_data["error"])
        else:
            # Create tabs for different sections
            price_tab, chart_tab, recommendations_tab, fundamentals_tab, news_tab = st.tabs([
                "Price Overview", "Price Chart", "Analyst Recommendations", "Fundamentals", "News"
            ])
            
            # Price Overview Tab
            with price_tab:
                current = price_data.get("current", {})
                
                # Company header with logo if available
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Try to get company logo
                    try:
                        stock = yf.Ticker(ticker)
                        logo_url = stock.info.get('logo_url')
                        if logo_url:
                            st.image(logo_url, width=100)
                    except:
                        pass
                
                with col2:
                    st.subheader(f"{current.get('name', 'Unknown')} ({current.get('symbol', ticker)})")
                    
                    # Current price and change
                    price = current.get('price', 'N/A')
                    change = current.get('change', 0)
                    
                    price_col, change_col = st.columns(2)
                    with price_col:
                        st.metric("Current Price", f"${price}" if isinstance(price, (int, float)) else str(price))
                    with change_col:
                        change_value = f"{change:.2f}%" if isinstance(change, (int, float)) else str(change)
                        st.metric("Change", change_value, delta=change_value)
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Previous Close", f"${current.get('previous_close', 'N/A')}")
                    st.metric("Day Low", f"${current.get('day_low', 'N/A')}")
                
                with col2:
                    st.metric("Open", f"${current.get('open', 'N/A')}")
                    st.metric("Day High", f"${current.get('day_high', 'N/A')}")
                
                with col3:
                    volume = current.get('volume', 'N/A')
                    volume_formatted = f"{volume:,}" if isinstance(volume, (int, float)) else str(volume)
                    st.metric("Volume", volume_formatted)
                
                with col4:
                    market_cap = fundamentals_data.get('market_cap', 'N/A')
                    market_cap_formatted = format_large_number(market_cap)
                    st.metric("Market Cap", market_cap_formatted)
            
            # Chart Tab
            with chart_tab:
                # Get the current stock data if not already defined
                current = price_data.get("current", {})
                
                # Time period selection with appropriate intervals
                period_intervals = {
                    "1M": ("1mo", "1d"),
                    "3M": ("3mo", "1d"),
                    "6M": ("6mo", "1d"),
                    "1Y": ("1y", "1d"),
                    "2Y": ("2y", "1wk"),
                    "5Y": ("5y", "1wk"),
                    "Max": ("max", "1mo")
                }
                period = st.selectbox("Select Time Period", list(period_intervals.keys()), index=0)
                
                # Get historical data with proper interval
                with st.spinner(f"Loading {period} data..."):
                    try:
                        stock = yf.Ticker(ticker)
                        period_code, interval = period_intervals[period]
                        hist_df = stock.history(period=period_code, interval=interval)
                        
                        if hist_df.empty:
                            st.warning("No data available for the selected period.")
                            return
                            
                        # Resample data for better visualization if needed
                        if len(hist_df) > 1000:
                            hist_df = hist_df.resample('D').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        
                        # Chart type selection
                        chart_type = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True)
                        
                        # Add moving averages selection
                        show_ma = st.checkbox("Show Moving Averages", value=False)
                        if show_ma:
                            ma_periods = st.multiselect(
                                "Select Moving Average Periods",
                                [5, 10, 20, 50, 100, 200],
                                default=[20, 50]
                            )
                            
                            # Calculate moving averages
                            for period in ma_periods:
                                if len(hist_df) >= period:
                                    hist_df[f'MA_{period}'] = hist_df['Close'].rolling(window=period).mean()
                        
                        if chart_type == "Line":
                            fig = go.Figure()
                            
                            # Add price line
                            fig.add_trace(
                                go.Scatter(
                                    x=hist_df.index,
                                    y=hist_df['Close'],
                                    name="Price",
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            # Add moving averages to line chart
                            if show_ma:
                                colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
                                for i, period in enumerate(ma_periods):
                                    if f'MA_{period}' in hist_df.columns:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=hist_df.index,
                                                y=hist_df[f'MA_{period}'],
                                                name=f'{period}-MA',
                                                line=dict(color=colors[i % len(colors)])
                                            )
                                        )
                            
                        else:  # Candlestick
                            fig = go.Figure()
                            
                            # Add candlestick
                            fig.add_trace(
                                go.Candlestick(
                                    x=hist_df.index,
                                    open=hist_df['Open'],
                                    high=hist_df['High'],
                                    low=hist_df['Low'],
                                    close=hist_df['Close'],
                                    name="Price"
                                )
                            )
                            
                            # Add moving averages to candlestick chart
                            if show_ma:
                                colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
                                for i, period in enumerate(ma_periods):
                                    if f'MA_{period}' in hist_df.columns:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=hist_df.index,
                                                y=hist_df[f'MA_{period}'],
                                                name=f'{period}-MA',
                                                line=dict(color=colors[i % len(colors)])
                                            )
                                        )
                        
                        # Update layout for both chart types
                        fig.update_layout(
                            title=f"{current.get('name', ticker)} Stock Price",
                            yaxis_title="Price ($)",
                            xaxis_title="Date",
                            template="plotly_white",
                            xaxis_rangeslider_visible=False,
                            height=600
                        )
                        
                        # Update y-axis format
                        fig.update_yaxes(tickprefix="$")
                        
                        # Show the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume chart
                        if 'Volume' in hist_df.columns and not hist_df['Volume'].isnull().all():
                            volume_fig = go.Figure(
                                go.Bar(
                                    x=hist_df.index,
                                    y=hist_df['Volume'],
                                    name="Volume"
                                )
                            )
                            volume_fig.update_layout(
                                title=f"{current.get('name', ticker)} Trading Volume",
                                yaxis_title="Volume",
                                xaxis_title="Date",
                                template="plotly_white",
                                height=300
                            )
                            st.plotly_chart(volume_fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading chart data: {str(e)}")
            
            # Recommendations Tab
            with recommendations_tab:
                if "message" in recommendations_data:
                    st.info(recommendations_data["message"])
                elif "error" in recommendations_data:
                    st.error(recommendations_data["error"])
                else:
                    recommendations_df = recommendations_data.get("recommendations_df")
                    
                    if recommendations_df is not None and not recommendations_df.empty:
                        # Summary metrics
                        st.subheader("Analyst Recommendations Summary")
                        
                        # Calculate recommendation distribution
                        buy_count = sum(1 for _, rec in recommendations_df.iterrows() if 'buy' in str(rec.get('To Grade', '')).lower())
                        sell_count = sum(1 for _, rec in recommendations_df.iterrows() if 'sell' in str(rec.get('To Grade', '')).lower())
                        hold_count = sum(1 for _, rec in recommendations_df.iterrows() if 'hold' in str(rec.get('To Grade', '')).lower() or 'neutral' in str(rec.get('To Grade', '')).lower())
                        total = len(recommendations_df)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Buy", f"{buy_count} ({buy_count/total*100:.1f}%)")
                        with col2:
                            st.metric("Hold", f"{hold_count} ({hold_count/total*100:.1f}%)")
                        with col3:
                            st.metric("Sell", f"{sell_count} ({sell_count/total*100:.1f}%)")
                        with col4:
                            st.metric("Total Ratings", str(total))
                        
                        # Recommendations pie chart
                        fig = px.pie(
                            values=[buy_count, hold_count, sell_count],
                            names=["Buy", "Hold", "Sell"],
                            title="Analyst Recommendations Distribution",
                            color_discrete_sequence=["#1cc88a", "#f6c23e", "#e74a3b"],
                            hole=0.4
                        )
                        st.plotly_chart(fig)
                        
                        # Recent recommendations table
                        st.subheader("Recent Recommendations")
                        
                        # Format the dataframe for display
                        display_df = recommendations_df.copy()
                        if not display_df.empty:
                            # Handle Date column format
                            if 'Date' in display_df.columns:
                                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                            
                            # Select and reorder columns
                            display_cols = ['Date', 'Firm', 'To Grade', 'From Grade', 'Action']
                            display_cols = [col for col in display_cols if col in display_df.columns]
                            
                            # Show the most recent 10 recommendations
                            st.dataframe(display_df[display_cols].head(10), use_container_width=True)
                    else:
                        st.info("No analyst recommendations available for this stock.")
            
            # Fundamentals Tab
            with fundamentals_tab:
                if "error" in fundamentals_data:
                    st.error(fundamentals_data["error"])
                else:
                    st.subheader("Key Financial Metrics")
                    
                    # Valuation metrics
                    st.markdown("#### Valuation")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Market Cap", format_large_number(fundamentals_data.get("market_cap", "N/A")))
                    with col2:
                        st.metric("Forward P/E", str(fundamentals_data.get("forward_pe", "N/A")))
                    with col3:
                        st.metric("Trailing P/E", str(fundamentals_data.get("trailing_pe", "N/A")))
                    with col4:
                        dividend_yield = fundamentals_data.get("dividend_yield", "N/A")
                        st.metric("Dividend Yield", format_percentage(dividend_yield))
                    
                    # Performance metrics
                    st.markdown("#### Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("52-Week High", f"${fundamentals_data.get('52_week_high', 'N/A')}")
                    with col2:
                        st.metric("52-Week Low", f"${fundamentals_data.get('52_week_low', 'N/A')}")
                    with col3:
                        st.metric("50-Day Avg", f"${fundamentals_data.get('50_day_average', 'N/A')}")
                    with col4:
                        st.metric("200-Day Avg", f"${fundamentals_data.get('200_day_average', 'N/A')}")
                    
                    # Financial Health
                    st.markdown("#### Financial Health")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Beta", str(fundamentals_data.get("beta", "N/A")))
                    with col2:
                        st.metric("Debt to Equity", str(fundamentals_data.get("debt_to_equity", "N/A")))
                    with col3:
                        profit_margins = fundamentals_data.get("profit_margins", "N/A")
                        st.metric("Profit Margins", format_percentage(profit_margins))
                    with col4:
                        revenue_growth = fundamentals_data.get("revenue_growth", "N/A")
                        st.metric("Revenue Growth", format_percentage(revenue_growth))
                    
                    # Cash Flow
                    st.markdown("#### Cash Flow")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Free Cash Flow", format_large_number(fundamentals_data.get("free_cash_flow", "N/A")))
                    with col2:
                        st.metric("Operating Cash Flow", format_large_number(fundamentals_data.get("operating_cash_flow", "N/A")))
                    with col3:
                        roe = fundamentals_data.get("return_on_equity", "N/A")
                        st.metric("Return on Equity", format_percentage(roe))
                    
                    # Check for quarterly financials
                    if "quarterly_financials_df" in fundamentals_data:
                        st.subheader("Quarterly Financials")
                        st.dataframe(fundamentals_data["quarterly_financials_df"], use_container_width=True)
            
            # News Tab
            with news_tab:
                if "message" in news_data:
                    st.info(news_data["message"])
                elif "error" in news_data:
                    st.error(news_data["error"])
                else:
                    articles = news_data.get("articles", [])
                    
                    if articles:
                        st.subheader(f"Recent News for {company_name}")
                        
                        # Display each news article
                        for i, article in enumerate(articles[:5]):
                            title = article.get('title', 'No Title')
                            source = article.get('source', {}).get('name', 'Unknown Source') if isinstance(article.get('source'), dict) else article.get('publisher', 'Unknown Source')
                            url = article.get('url', '#')
                            published_at = article.get('publishedAt', article.get('providerPublishTime', 'Unknown Date'))
                            
                            # Format timestamp if it's a timestamp
                            if isinstance(published_at, int):
                                published_at = datetime.fromtimestamp(published_at).strftime('%Y-%m-%d %H:%M')
                                
                            description = article.get('description', 'No description available.')
                            
                            with st.expander(f"{title}", expanded=i==0):
                                st.markdown(f"**Source:** {source} - {published_at}")
                                st.markdown(description)
                                st.markdown(f"[Read more]({url})")
                                st.markdown("---")
                    else:
                        st.info(f"No recent news found for {company_name}")

if __name__ == "__main__":
    main()