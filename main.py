import os

import torch
from diffusers import FluxPipeline
import re

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def sanitize_filename(text):
    # Replace spaces and special characters with underscores
    safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', text)
    # Truncate to 32 characters
    safe_filename = safe_filename[:32]
    return safe_filename


def app():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell",
                                        torch_dtype=torch.bfloat16).to(DEVICE)
    # pipe.enable_model_cpu_offload()

    print("Type 'quit' to exit.")
    while True:
        prompt = input("> ")
        if prompt == "quit":
            break
        image = pipe(
            prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=4,
            # max_sequence_length=256,
            # generator=torch.Generator(DEVICE).manual_seed(0)
        ).images[0]
        base_file_name = sanitize_filename(prompt)
        count = 0
        file_name = base_file_name
        while os.path.exists(file_name + ".png"):
            file_name = base_file_name + "." + str(count)
            count = count + 1
        file_name = file_name + ".png"
        image.save(file_name)


if __name__ == "__main__":
    app()
