from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


def image_description_generator(image_paths):
    """Caption generator for a list of images."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "Salesforce/blip2-opt-6.7b"  # 2.7b 6.7b 2.7b-coco 6.7b-coco
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)

    generated_captions = []
    for path in image_paths:
        image = Image.open(path)
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        generated_text += '.' if generated_text[-1] != '.' else ''
        generated_captions.append(generated_text)

    return generated_captions


if __name__ == "__main__":
    base_dir = "./datasets/image/"
    image_paths = [base_dir + "seq_eth_reference.png", 
                   base_dir + "seq_hotel_reference.png",
                   base_dir + "students003_reference.png",
                   base_dir + "crowds_zara01_reference.png",
                   base_dir + "crowds_zara02_reference.png"]
    caption_paths = [base_dir + "seq_eth_caption.txt", 
                     base_dir + "seq_hotel_caption.txt",
                     base_dir + "students003_caption.txt",
                     base_dir + "crowds_zara01_caption.txt",
                     base_dir + "crowds_zara02_caption.txt"]
    
    # Generate captions
    captions = image_description_generator(image_paths)
    print(list(zip(image_paths, captions, caption_paths)))

    # Save captions to file
    for i, path in enumerate(caption_paths):
        with open(path, "w") as f:
            f.write(captions[i])
