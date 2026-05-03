import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "",
    model="tencent/HunyuanImage-3.0",
)

image.show()
image.save("astronaut_riding_a_horse.png")