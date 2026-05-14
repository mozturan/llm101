import os
from dotenv import load_dotenv
from google import genai
import json

# Load variables from .env into the environment
load_dotenv()

# Access the key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API Key not found. Did you set it in the .env file?")

client = genai.Client(api_key=api_key)

# 1. DEFINE TOOLS
def get_stock_price(ticker):
    """Returns the mock stock price for a given ticker."""
    prices = {"AAPL": 150, "TSLA": 200, "GOOG": 2800}
    return f"The price of {ticker} is ${prices.get(ticker, 'unknown')}"

def multiply(a, b):
    """Multiplies two numbers."""
    return f"Result: {float(a) * float(b)}"

def run_python(code: str) -> str:
    """Execute Python code safely and return output."""
    import io
    import contextlib
    
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})
        return output.getvalue() or "Code executed successfully, no output."
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Map tool names to actual functions
TOOLS = {
    "get_stock_price": get_stock_price,
    "multiply": multiply,
    "run_python": run_python,
    "get_current_time": get_current_time
}

# Tool descriptions for the LLM
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the for the requested ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker name"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return the output. Use for calculations, data processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Get multiplication of two values",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "float", "description": "First value for multiply function"},
                    "b": {"type": "float", "description": "Second value for multiply function"}
                },
                "required": ["a", "b"]
            }
        }
    },
]


# --- The Agent Loop ---

def run_agent(user_goal: str, max_iterations: int = 10):
    
    system_prompt = """You are a helpful AI agent. You have access to tools.
Use them step by step to achieve the user's goal.
Think carefully about which tool to use and why.
When you have enough information, provide a final answer."""
    
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI agent. You have access to tools.
Use them step by step to achieve the user's goal.
Think carefully about which tool to use and why.
When you have enough information, provide a final answer."""
        },
        {
            "role": "user",
            "content": user_goal
        }
    ]


system_prompt = "Sen insanların sorularına cevap veren bilge birisin ve Yoda gibi konuşuyorsun."
user_prompt = "Hayatın anlamını nasıl bulabilirim?"

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
        config={
            "system_instruction": system_prompt
        },
        contents=[user_prompt]
)

print(response.text)