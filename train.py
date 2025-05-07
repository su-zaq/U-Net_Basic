import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNet
from loss import DiceLoss
from dataset import SegmentationDataset
from utils import save_model
from send_info_discord import discord_info

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = SegmentationDataset("./dataset/images/", "./dataset/masks/", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = nn.DataParallel(UNet()).to(device)  # 複数GPU対応
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 40

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    save_model(model.module, "unet_model.pth")  # .module でUNet本体を保存
    print("Model saved!")
    discord_info(117)


if __name__ == "__main__":
    main()