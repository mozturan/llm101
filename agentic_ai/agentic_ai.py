import argparse
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os


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

def get_files_info(working_directory: str, directory: str = ".") -> str:
    """List files in a directory and return their names and sizes."""
    abs_working_directory = os.path.abspath(working_directory)
    if directory == ".":
        target_directory = abs_working_directory
    else:
        target_directory = os.path.join(abs_working_directory, directory)

    if not os.path.exists(target_directory):
        return f"Error: Directory '{directory}' does not exist."
    
    if not os.path.isdir(target_directory):
        return f"Error: '{directory}' is not a directory."

    entries_info = []
    for filename in os.listdir(target_directory):
        file_path = os.path.join(target_directory, filename)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            entries_info.append(f"{filename} ({size} bytes)")
        elif os.path.isdir(file_path):
            entries_info.append(f"{filename}/ (directory)")
    
    return "Entries in directory:\n" + "\n".join(entries_info) if entries_info else "No files or directories found in the directory."

def get_file_content(working_directory: str, file_path: str) -> str:
    """Read and return the content of a specified file."""
    abs_working_directory = os.path.abspath(working_directory)
    target_file = os.path.join(abs_working_directory, file_path)

    if not os.path.exists(target_file):
        return f"Error: File '{file_path}' does not exist."
    
    if not os.path.isfile(target_file):
        return f"Error: '{file_path}' is not a file."
    max_size = 5 * 1024 * 1024  # 5 MB
    if os.path.getsize(target_file) > max_size:
        return f"Error: File '{file_path}' exceeds the maximum allowed size of 5 MB."
    
    max_characters = 10000
    try:
        with open(target_file, 'r') as f:
            content = f.read(max_characters)
            if len(content) == max_characters:
                return f"Error: File '{file_path}' exceeds the maximum allowed character limit of {max_characters}."
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(working_directory: str, file_path: str, content: str) -> str:
    """Write content to a specified file."""
    abs_working_directory = os.path.abspath(working_directory)
    target_file = os.path.join(abs_working_directory, file_path)

    # Ensure the target file is within the working directory
    if not os.path.commonpath([abs_working_directory, target_file]) == abs_working_directory:
        return "Error: Invalid file path. Access denied."

    try:
        with open(target_file, 'w') as f:
            f.write(content)
        return f"Content successfully written to '{file_path}'."
    except Exception as e:
        return f"Error writing to file: {str(e)}"
    

# Map tool names to actual functions
TOOLS = {
    "get_stock_price": get_stock_price,
    "multiply": multiply,
    "run_python": run_python,
    "get_current_time": get_current_time,
    "get_files_info": get_files_info,
    "get_file_content": get_file_content,
    "write_file": write_file
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
        ),
        types.FunctionDeclaration(
            name="get_files_info",
            description="List files in a directory and return their names and sizes.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "working_directory": types.Schema(type=types.Type.STRING, description="The base directory for file operations"),
                    "directory": types.Schema(type=types.Type.STRING, description="Target directory to list files from, relative to working_directory. Use '.' for the working directory.")
                },
                required=["working_directory", "directory"]
            )
        ),
        types.FunctionDeclaration(
            name="get_file_content",
            description="Read and return the content of a specified file.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "working_directory": types.Schema(type=types.Type.STRING, description="The base directory for file operations"),
                    "file_path": types.Schema(type=types.Type.STRING, description="Path to the file to read, relative to working_directory")
                },
                required=["working_directory", "file_path"]
            )
        ),
        types.FunctionDeclaration(
            name="write_file",
            description="Write content to a specified file.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "working_directory": types.Schema(type=types.Type.STRING, description="The base directory for file operations"),
                    "file_path": types.Schema(type=types.Type.STRING, description="Path to the file to write, relative to working_directory"),
                    "content": types.Schema(type=types.Type.STRING, description="Content to write to the file")
                },
                required=["working_directory", "file_path", "content"]
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

def run_agent(user_goal: str, max_iterations: int = 10, verbose: bool = False):
    
    print(f"User Goal: {user_goal}\n")

    # Conversation history - managed manually
    contents: list = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_goal)]
        )]


    for iteration in range(max_iterations):
        if verbose:
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
                parts=candidate.content.parts
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
        
        # If there are tool calls — execute them
        if tool_calls:
            tool_result_parts = []
            
            for tool_call in tool_calls:
                tool_name = tool_call.name
                # Convert args to plain dict
                tool_args = dict(tool_call.args) if tool_call.args else {}
                
                if verbose:
                    print(f"Tool called: {tool_name}")
                    print(f"Arguments: {tool_args}")
                    
                # Execute the actual function
                if tool_name in TOOLS:
                    if tool_args:
                        result = TOOLS[tool_name](**tool_args)
                    else:
                        result = TOOLS[tool_name]()
                else:
                    result = f"Error: tool '{tool_name}' not found"
                
                if verbose:
                    print(f"Result: {result}\n")
                
                # Build function response — include id for Gemini 3 models
                tool_result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=tool_call.id,    # Required for Gemini 3 models
                            name=tool_name,
                            response={"result": result}
                        )
                    )
                )
            
            # Add all tool results to history as a user turn
            contents.append(
                types.Content(
                    role="user",
                    parts=tool_result_parts
                )
            )
        
        else:
            # No tool calls — this is the final answer
            final_answer = "\n".join(text_parts) if text_parts else "No response generated."
            print("\n=== FINAL ANSWER ===")
            print(final_answer)

            if verbose:
                print(f"Total iterations: {iteration + 1}")
                print("\n====================")
                print(f"Prompt tokens used: {response.usage_metadata.prompt_token_count}")
                print(f"Generation tokens used: {response.usage_metadata.candidates_token_count}")

            return final_answer
    
    return "Max iterations reached without final answer."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose debug output")

    args = parser.parse_args()

    # prompt_ = "What is the current price of AAPL stock and what is 12 multiplied by 15?"

    if not args.user_prompt:
        print("Please provide a user prompt.")
    else:
        run_agent(args.user_prompt, verbose=args.verbose)