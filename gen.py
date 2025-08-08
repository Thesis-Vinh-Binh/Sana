import time
import torch
import sys
sys.path.append("/workspace")

from PIL import Image
from torchvision.transforms.functional import to_pil_image
from Sana.app.sana_controlnet_pipeline import SanaControlNetPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = SanaControlNetPipeline("/workspace/Sana/configs/sana_controlnet_config/Sana_1600M_1024px_controlnet_bf16.yaml")
pipe.from_pretrained("hf://Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED/checkpoints/Sana_1600M_1024px_BF16_ControlNet_HED.pth")

pipe.to(device)

def generate(img_path, output_path, prompt):
    ref_image = Image.open(img_path).convert("RGB")
    seed = int(time.time())  # hoặc random.randint(...)

    output = pipe(
        prompt=prompt,
        ref_image=ref_image,
        guidance_scale=5.0,
        num_inference_steps=10,
        sketch_thickness=2,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # Nếu output là tensor, convert nó sang PIL Image
    if isinstance(output, torch.Tensor):
        # Nếu batch, chọn ảnh đầu tiên
        image_tensor = output[0].detach().cpu().to(torch.float32)
        # Scale từ [-1, 1] sang [0, 1] nếu cần
        image_tensor = (image_tensor + 1) / 2
        image_pil = to_pil_image(image_tensor)
        image_pil.save(output_path)
    elif isinstance(output, list) and isinstance(output[0], Image.Image):
        output[0].save(output_path)
    else:
        raise ValueError("Unknown output type from pipeline:", type(output))

if __name__ == "__main__":
    print("Generating image with Sana ControlNet...")
    # img_path = "asset/controlnet/ref_images/A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a la.jpg"
    # prompt = "A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a landscape."
    img_path = "/workspace/evaluate-2d/dataset/experiment/testset/000_keepcurve_botleft_44.png"
    prompt = "Complete the clothes image without human face. Follow caption detail" + '''
    The image shows a **long-sleeved, V-neck sweater** with a distinctive and colorful design. Here's a detailed description:

**Overall Style:** It has a relaxed, slightly oversized fit, suggesting a comfortable and casual style. The V-neck adds a touch of sophistication.

**Color and Pattern:** The sweater is primarily **off-white or cream** with a textured **cable knit pattern** throughout the main body. The sleeves and a portion of the front panels feature
    '''
    generate(img_path, "output/gen.png", prompt)
