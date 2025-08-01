# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)
from safetensors.torch import load_file

lora_state_dict = load_file("/workspace/xyzsketchstylev2-000002.safetensors")
pipe.load_lora_weights(
    "/workspace/xyzsketchstylev2-000002.safetensors",
    state_dict=lora_state_dict,
    adapter_name="xyzsketchstyle"
)
# pipe.set_adapters(["xyzsketchstyle"], adapter_weights=[2.0])

prompt = "a full body dog, intricate sketch style, detailed design"
# print(pipe.adapters)  # Should show that 'xyzsketchstyle' is active

# prompt = 'a winter coat uniform for woman, in xyzsketchstyle style, intricate details<lora:xyzsketchstylev2reg-000002:2.0>'
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]

image[0].save("dog_sketch.png")
