from google import genai
from google.genai import types

# 1. DEFINE YOUR TOOL
def get_stock_price(ticker: str):
    """Returns the current price of a stock ticker."""
    # Mock data for demonstration
    prices = {"AAPL": 150.50, "GOOGL": 2800.10, "NVDA": 950.00}
    return {"price": prices.get(ticker.upper(), "Ticker not found")}

# 2. INITIALIZE CLIENT
client = genai.Client(api_key='YOUR_API_KEY')
MODEL_NAME = "gemini-3.1-flash-lite"

def run_agentic_loop(prompt):
    # Initialize history with the user's goal
    messages = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    
    print(f"--- MISSION START: {prompt} ---")

    # Limit loops (e.g., 5) to prevent runaway costs/logic
    for i in range(5):
        # Step A: Request thought/action from Gemini
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=messages,
            config=types.GenerateContentConfig(
                tools=[get_stock_price],
                system_instruction="You are a financial agent. Use tools to find facts."
            )
        )

        # Step B: Record the model's response (its 'Thought') in history
        model_content = response.candidates[0].content
        messages.append(model_content)

        # Step C: Extract Function Calls
        function_calls = [p.function_call for p in model_content.parts if p.function_call]

        if not function_calls:
            # No more tools needed; process the final text answer
            print(f"\n[FINAL ANSWER]: {response.text}")
            break

        # Step D: Execute the Tools and map to IDs
        response_parts = []
        for fc in function_calls:
            print(f"  [Loop {i+1}]: AI calling {fc.name} with {fc.args}")
            
            # Execute the function
            result = get_stock_price(**fc.args)

            # Step E: Create a response part that includes the CALL ID
            # Gemini 3.1 REQUIRES the 'id' to be returned
            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response=result
                )
            )

        # Append the tool results to the conversation history
        messages.append(types.Content(role="user", parts=response_parts))

run_agentic_loop("I want to buy 5 shares of AAPL. How much will that cost?")