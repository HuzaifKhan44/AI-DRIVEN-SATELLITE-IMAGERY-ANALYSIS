import os
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import PatchDataset
from model import UNet  # Ensure UNet is defined correctly

# Dice Coefficient
def dice_coefficient(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, writer, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    train_losses = []
    val_losses = []
    val_dices = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)

            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        dice_scores = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)

                loss = criterion(preds, masks)
                val_loss += loss.item()

                dice = dice_coefficient(preds, masks)
                dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = sum(dice_scores) / len(dice_scores)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_dice)

        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Dice/Val", avg_dice, epoch)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Dice = {avg_dice:.4f}")

    return train_losses, val_losses, val_dices, model

# Plot losses
def plot_losses(train_losses, val_losses, save_path="training_plot.png"):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Loss plot saved to {save_path}")

# Main function
def main():
    dataset = PatchDataset("data/patches")  # Update path if needed
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = UNet(in_channels=3, out_channels=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    writer = SummaryWriter("runs/unet_training")

    train_losses, val_losses, val_dices, trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, writer, num_epochs=10
    )

    writer.close()
    os.makedirs("models", exist_ok=True)
    torch.save(trained_model.state_dict(), "models/unet_trained.pth")
    print("[INFO] Model saved to models/unet_trained.pth")

    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()
