import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()
# Load variables from .env into the environment
load_dotenv()

# Access the key
api_key = os.getenv("HF_TOKEN")

if not api_key:
    raise ValueError("API Key not found. Did you set it in the .env file?")

client = InferenceClient(
    api_key=api_key
)

# Example: Using an ethical, efficient model like Mistral-7B
messages = [{"role": "user", 
             "content": "How can I make an NPC in a 2D indie game feel more alive?"}]

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:groq",
    messages=messages,
)

message = completion.choices[0].message.content
print("\n\n")
md = Markdown(message or "")
console.print(Panel(md, title="AI Response", expand=False), soft_wrap=True)
