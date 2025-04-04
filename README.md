# Stock Analysis Dashboard

An AI-powered stock analysis dashboard built with Streamlit that provides real-time stock data, technical analysis, and news updates. The dashboard uses an intelligent financial agent to analyze and present comprehensive stock information.

## AI Features

- Intelligent data aggregation from multiple sources
- Smart technical analysis with automated pattern recognition
- Dynamic news relevance filtering
- Adaptive data visualization based on stock characteristics
- Automated fundamental analysis with key metrics highlighting

## Features

- Real-time stock price tracking
- Interactive price charts (Line and Candlestick)
- Moving average technical indicators
- Analyst recommendations analysis
- Company fundamentals and financial metrics
- Latest company news integration
- Multiple timeframe analysis

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`
- News API key (optional, for enhanced news features)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd financial-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your News API key (optional):
     ```
     NEWS_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit app:
```bash
streamlit run stockagent.py
```

2. Enter a stock ticker symbol in the sidebar
3. Click "Analyze Stock" to view comprehensive analysis

## Data Sources

- Stock data: Yahoo Finance (yfinance)
- News data: News API (if configured) with fallback to Yahoo Finance news

## Components

1. **Price Overview**
   - Current price and daily change
   - Key market metrics
   - Company logo and basic info

2. **Price Chart**
   - Interactive price charts
   - Multiple timeframe options
   - Moving average overlays
   - Volume analysis

3. **Analyst Recommendations**
   - Buy/Sell/Hold distribution
   - Recent analyst ratings
   - Visual recommendation summary

4. **Fundamentals**
   - Key financial metrics
   - Performance indicators
   - Valuation metrics
   - Financial health indicators

5. **News Feed**
   - Latest company news
   - News source attribution
   - Direct links to full articles

## Contributing

Feel free to fork the project and submit pull requests with improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
