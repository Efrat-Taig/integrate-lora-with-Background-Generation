"""
This script demonstrates the integration of a LoRA (Low-Rank Adaptation) model with a background generation model.
It processes images by replacing their backgrounds using two methods:
1. The original BRIA-2.3 model without LoRA
2. The BRIA-2.3 model with a LoRA model trained on modern, blurred sea views

The script showcases how the LoRA model influences the background generation process,
adapting it to the style learned from the training dataset.

Key steps:
1. Load necessary models and components
2. Process images using the original model
3. Process images using the model with LoRA weights
4. Save the results for comparison

Requirements:
- BRIA-2.3 model and its ControlNet
- A trained LoRA model
- Input images in the specified input directory
"""

import os
import torch
import gc
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, LCMScheduler
from replace_bg.model.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from replace_bg.model.controlnet import ControlNetModel
from replace_bg.utilities import resize_image, remove_bg_from_image, paste_fg_over_image, get_control_image_tensor
from PIL import Image

# Function to clear GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Function to process a single image
def process_image(image_path, prompt, negative_prompt, pipe, num_inference_steps=50):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = resize_image(image)
    mask = remove_bg_from_image(image_path)
    control_tensor = get_control_image_tensor(pipe.vae, image, mask)

    # Generate new background
    seed = 0
    generator = torch.Generator("cuda").manual_seed(seed)
    gen_img = pipe(
        negative_prompt=negative_prompt,
        prompt=prompt,
        controlnet_conditioning_scale=1.0,
        num_inference_steps=num_inference_steps,
        image=control_tensor,
        generator=generator
    ).images[0]

    # Combine new background with original foreground
    result_image = paste_fg_over_image(gen_img, image, mask)
    return result_image

# Function to process all images in a directory
def process_all_images(pipe, input_dir, output_dir, prompt, negative_prompt, prefix, num_inference_steps=50):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            result_image = process_image(input_path, prompt, negative_prompt, pipe, num_inference_steps)
            output_path = os.path.join(output_dir, f"{prefix}_{filename}")
            result_image.save(output_path)
            print(f"Saved {output_path}")

# Define directories
input_dir = "/data/input"
output_dir = "/data/output"
os.makedirs(output_dir, exist_ok=True)

# Define prompt and negative prompt
prompt = "A distant, tranquil view of the ocean at sunrise, seen from a high-rise window with soft, warm reflections. Aesthetic background for profile picture."
negative_prompt = "floor, ceiling, walls, furniture, people, sharp details, high contrast, vibrant colors"

# Load shared components
controlnet = ControlNetModel.from_pretrained("briaai/BRIA-2.3-ControlNet-BG-Gen", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Process with original model
clear_gpu_memory()
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("briaai/BRIA-2.3", controlnet=controlnet, torch_dtype=torch.float16, vae=vae).to('cuda:0')
pipe.scheduler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    num_train_timesteps=1000, steps_offset=1
)
process_all_images(pipe, input_dir, output_dir, prompt, negative_prompt, "original")
del pipe
clear_gpu_memory()

# Process with Lora Beer
clear_gpu_memory()
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("briaai/BRIA-2.3", controlnet=controlnet, torch_dtype=torch.float16, vae=vae).to('cuda:0')
pipe.scheduler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    num_train_timesteps=1000, steps_offset=1
)
lora_model_path = "Modern_Blurred_SeaView/checkpoint-1400/pytorch_lora_weights.safetensors"
pipe.load_lora_weights(lora_model_path)
checkpoint_num = 1400
in_prefix = f'Modern_Blurred_SeaView_{checkpoint_num}'
process_all_images(pipe, input_dir, output_dir, prompt, negative_prompt, in_prefix)
del pipe
clear_gpu_memory()

print("All images processed.")
