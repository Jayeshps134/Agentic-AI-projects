# -----------------------------------------------------
# imports
# -----------------------------------------------------
import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from datetime import date, timedelta
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------------
# Tools
# -----------------------------------------------------

# inbuilt tools
duck_search = DuckDuckGoSearchRun()
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# custom tool : stock price search
class StockInput(BaseModel):
    ticker_name: str = Field(
        ...,
        description="Stock ticker symbol used by Yahoo Finance. Use the exchange-specific ticker when required, e.g. AAPL for Apple (US), RELIANCE.NS for Reliance Industries (NSE India), or TCS.NS for TCS (NSE India)."
    )
    Date: date = Field(
        default_factory=date.today,
        description="The trading date for which the historical closing price is requested. Use the date in YYYY-MM-DD format. If the user does not specify a date, use today's date."
    )
class StockOutput(BaseModel):
    ticker_name: str = Field(
        ...,
        description="The stock ticker symbol used to retrieve the historical price."
    )
    Date: date = Field(
        ...,
        description="The date for which the historical stock price was retrieved, in YYYY-MM-DD format."
    )
    closing_Price: float = Field(
        ...,
        description="The stock's closing market price on the specified trading date."
    )

@tool(args_schema=StockInput)
def StockPriceSearch(ticker_name: str, Date: date) -> StockOutput:
    """Get the historical closing price of a stock for a specific trading date.
    Use this tool when the user asks for the stock price or closing price
    of a particular company on a specific date.
    The ticker_name must be a Yahoo Finance ticker symbol.
    The Date specifies the trading date for which the closing price is required.
    If the specified date is not a trading day, such as a weekend or market holiday,
    no historical price may be available.
    """
    ticker_name = ticker_name.upper()
    stock_data = yf.Ticker(ticker_name).history(start=Date, end=Date + timedelta(days=1))

    if stock_data.empty:
        raise ValueError(f"No historical closing price found for {ticker_name} on {Date}. The date may be a weekend, market holiday, or invalid trading date.")

    return StockOutput(
        ticker_name=ticker_name,
        Date=Date,
        closing_Price=float(stock_data["Close"].iloc[0])
    )

available_tools = [duck_search, wiki_search, StockPriceSearch]