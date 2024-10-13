# Based on https://huggingface.co/briaai/BRIA-2.3-ControlNet-BG-Gen


import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from replace_bg.model.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from replace_bg.model.controlnet import ControlNetModel
from replace_bg.utilities import resize_image, remove_bg_from_image, paste_fg_over_image, get_control_image_tensor

# Load models
controlnet = ControlNetModel.from_pretrained("briaai/BRIA-2.3-ControlNet-BG-Gen", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("briaai/BRIA-2.3", controlnet=controlnet, torch_dtype=torch.float16, vae=vae).to('cuda:0')

# Set up scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    steps_offset=1
)

# Load and process image
image_path = "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg"
image = load_image(image_path)
image = resize_image(image)
mask = remove_bg_from_image(image_path)
control_tensor = get_control_image_tensor(pipe.vae, image, mask)

# Generate new background
prompt = "in a zoo"
negative_prompt = "Logo,Watermark,Text,Ugly,Bad proportions,Bad quality,Out of frame,Mutation"
generator = torch.Generator(device="cuda:0").manual_seed(0)

gen_img = pipe(
    negative_prompt=negative_prompt,
    prompt=prompt,
    controlnet_conditioning_scale=1.0,
    num_inference_steps=50,
    image=control_tensor,
    generator=generator
).images[0]

# Combine original foreground with new background
result_image = paste_fg_over_image(gen_img, image, mask)

# Save the result
result_image.save("result.png")

