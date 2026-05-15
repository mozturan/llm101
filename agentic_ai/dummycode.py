import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key not found. Did you set it in the .env file?")


# --- Init Client ---

client = genai.Client(api_key=api_key)
model = "gemini-3.1-flash-lite"


# --- DEFINE TOOLS ---
def get_stock_price(ticker) -> str:
    """Returns the mock stock price for a given ticker."""
    prices = {"AAPL": 150, "TSLA": 200, "GOOG": 2800}
    return f"The price of {ticker} is ${prices.get(ticker, 'unknown')}"

def multiply(a, b) -> str:
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
TOOL_DECLARATIONS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_stock_price",
            description="Get the price for the requested ticker",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ticker": types.Schema(type=types.Type.STRING, description="Ticker name")
                },
                required=["ticker"]
            )
        ),
        types.FunctionDeclaration(
            name="run_python",
            description="Execute Python code and return the output. Use for calculations, data processing.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(type=types.Type.STRING, description="Python code to execute")
                },
                required=["code"]
            )
        ),
        types.FunctionDeclaration(
            name="get_current_time",
            description="Get the current date and time",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={}
            )
        ),
        types.FunctionDeclaration(
            name="multiply",
            description="Get multiplication of two values",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "a": types.Schema(type=types.Type.NUMBER, description="First value for multiply function"),
                    "b": types.Schema(type=types.Type.NUMBER, description="Second value for multiply function")
                },
                required=["a", "b"]
            )
        )
    ]
)



# --- Config — system instruction + tools defined once ---

CONFIG = types.GenerateContentConfig(
    system_instruction="""You are a helpful AI agent. 
    You have access to tools.
    Use them step by step to achieve the user's goal. 
    Think carefully about which tool to use and why. 
    When you have enough information, provide a final answer.""",
    tools=[TOOL_DECLARATIONS]
)



# --- The Agent Loop ---

def run_agent(user_goal: str, max_iterations: int = 10):
    
    print(f"User Goal: {user_goal}\n")

    # Conversation history - managed manually
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_goal)]
        )]


    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")

        # Generate response from the model
        response = client.models.generate_content(
            model=model,
            config=CONFIG,
            contents=contents
        )

        candidate = response.candidates[0]
        # Add model's response to history
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part(text=candidate.content.parts)]
            )
        )

        # Separate tool calls from text parts
        tool_calls = []
        text_parts = []

        for part in candidate.content.parts:
            if part.function_call:
                tool_calls.append(part.function_call)
            else:
                text_parts.append(part.text)
        
        # Process tool calls
        for tool_call in tool_calls:
            tool_name = tool_call.name
            arguments = json.loads(tool_call.arguments)

            if tool_name in TOOLS:
                print(f"Calling tool: {tool_name} with arguments {arguments}")
                result = TOOLS[tool_name](**arguments)
                print(f"Tool result: {result}")

                # Add tool result to conversation history
                contents.append(
                    types.Content(
                        role="tool",
                        parts=[types.Part(text=result)]
                    )
                )
            else:
                print(f"Unknown tool: {tool_name}")

    pass

run_agent("What is the current price of AAPL stock and what is 12 multiplied by 15?")