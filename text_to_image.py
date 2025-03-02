from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate an image from text
def generate_image(prompt):
    print(f"Generating image for: '{prompt}'")
    image = pipe(prompt).images[0]
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")

# Input prompt from the user
if __name__ == "__main__":
    prompt = input("Enter a description for the image you want to generate: ")
    generate_image(prompt)
