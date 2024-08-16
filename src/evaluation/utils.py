import base64

import PIL.Image

from src.utils import check_hf_model_exists
from src.vlmevalkit_utils import is_vlmeval_models


def encode_image_base64(image: PIL.Image.Image):
    image.save("tmp", format=image.format)
    
    with open("tmp", "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_model_input(model_name: str, prompt: str, image: PIL.Image.Image):
    if "gpt" in model_name or "claude" in model_name or "Qwen2-VL" in model_name:
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        if image is not None:
            base64_image = encode_image_base64(image)

            if "gpt" in model_name:
                content.insert(
                    0,
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image.format.lower()};base64,{base64_image}",
                            "detail": "high",
                        }
                    }
                )
            elif "claude" in model_name:
                content.insert(
                    0,
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image.format.lower()}",
                            "data": base64_image,
                        },
                    }
                )
            elif "Qwen2-VL" in model_name:
                content.insert(
                    0,
                    {
                        "type": "image",
                        "image": f"data:image/{image.format.lower()};base64,{base64_image}",
                    }
                )
        
        return [
            {
                "role": "user",
                "content": content
            }
        ]
    
    if "gemini" in model_name:
        if image is None:
            return prompt
        else:
            return [image, prompt]
    
    if is_vlmeval_models(model_name):
        return prompt
    
    if check_hf_model_exists(model_name):
        return prompt

    raise NotImplementedError(f"Model {model_name} is not supported")

