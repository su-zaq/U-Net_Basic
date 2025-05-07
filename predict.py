import torch
from torchvision import transforms
from PIL import Image
import os
from model import UNet

#python predict_batch.py --input_dir test_images/ --output_dir predicted_masks/ --model unet_model.pth


def load_model(model_path, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze().cpu()

    mask = (output > 0.5).float() * 255
    mask_img = Image.fromarray(mask.byte().numpy())
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return mask_img

def batch_predict(input_dir, output_dir, model_path):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_mask.png")

        mask_img = predict_image(model, image_path, device)
        mask_img.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="画像フォルダのパス")
    parser.add_argument("--output_dir", required=True, help="マスク保存フォルダのパス")
    parser.add_argument("--model", default="unet_model.pth", help="学習済みモデルのパス")
    args = parser.parse_args()

    batch_predict(args.input_dir, args.output_dir, args.model)
