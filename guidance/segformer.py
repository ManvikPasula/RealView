# implementation for https://github.com/NVlabs/SegFormer

from google.colab import drive
drive.mount('/content/drive')
 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
 
DATASET_PATH = "/content/drive/MyDrive/Research/RealView/seg_dataset"
MODEL_SAVE_PATH = "/content/drive/MyDrive/Research/RealView/segformer/model_paths/"
CHECKPOINT_PATH = None  
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 1e-5
IMG_SIZE = (640, 480)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_INTERVAL = 5
 
 
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.processor = processor
        self.image_filenames = os.listdir(image_dir)
 
    def __len__(self):
        return len(self.image_filenames)
 
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace(".png", "_mask.png"))
 
       
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  
 
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
 
        image = np.array(image)
        mask = np.array(mask)
 
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
 
        mask = torch.tensor(mask, dtype=torch.long)  
        return pixel_values, mask
 
 
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
 
train_dataset = SegmentationDataset(
    os.path.join(DATASET_PATH, "train/images"),
    os.path.join(DATASET_PATH, "train/masks"),
    processor=image_processor
)
valid_dataset = SegmentationDataset(
    os.path.join(DATASET_PATH, "valid/images"),
    os.path.join(DATASET_PATH, "valid/masks"),
    processor=image_processor
)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
 
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1, alpha=0.5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
 
    def forward(self, inputs, targets):
        bce_score = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)  
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() + self.smooth
        dice = (2. * intersection + self.smooth) / union
 
        return self.alpha * bce_score + (1-self.alpha) * (1 - dice)  
 
 
def load_segformer(num_classes=1, device="cuda"):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    return model
 
model = load_segformer()
 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = DiceBCELoss(0.3)
 
def train_model(patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_loss_history, val_loss_history = [], []
 
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.long)
            masks = masks / 255
            optimizer.zero_grad()
 
            outputs = model(pixel_values=images).logits  
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs.squeeze(1), masks.float())  
 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
 
       
        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_train_loss:.4f}")
 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(valid_loader):
                images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.long)
                masks = masks / 255
                outputs = model(pixel_values=images).logits
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(outputs.squeeze(1), masks.float())
 
                val_loss += loss.item()
 
       
        avg_val_loss = val_loss / len(valid_loader)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
 
       
       
       
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print("ruh roh! It didn't get better!")
            if epochs_no_improve >= patience:
                print(" Early stopping triggered!")
                break
 
       
       
       
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"segformer_checkpoint_epoch{epoch+1}.pth"))
 
    return train_loss_history, val_loss_history
 
def infer_and_display(model, image_path, mask_path, device):
    image = Image.open(image_path).convert("RGB").resize((640, 480))
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
 
    with torch.no_grad():
        output = model(image_tensor).logits
        pred_mask = torch.nn.functional.interpolate(output, size=torch.Size([480, 640]), mode="bilinear", align_corners=False)
        pred_mask = (pred_mask.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
 
    if mask_path:
        mask = Image.open(mask_path).convert("L").resize((640, 480))  
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
    else:
        axes[1].set_title("No Mask Provided")
 
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    plt.show()
 
 
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "segformer_checkpoint_epoch15.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
 
 
test_image_path = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/images/2_side_test_3.png"
test_mask_path = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/masks/2_side_test_3_mask.png"
 
infer_and_display(model, test_image_path, test_mask_path, DEVICE)
 
def calculate_mean_iou(model, dataset, device):
   
    model.eval()
    iou_scores = []
 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
 
    with torch.no_grad():
        for images, true_masks in tqdm(dataloader, desc="Calculating IoU"):
            images = images.to(device)
            true_masks = true_masks.to(device) / 255
 
           
            outputs = model(pixel_values=images).logits
            pred_masks = torch.nn.functional.interpolate(
                outputs, size=true_masks.shape[-2:], mode="bilinear", align_corners=False
            )
            pred_masks = (torch.sigmoid(pred_masks).squeeze(1) > 0.5).float()
 
           
            intersection = torch.sum(pred_masks * true_masks)
            union = torch.sum(pred_masks) + torch.sum(true_masks) - intersection
 
            if union > 0:
                iou = intersection / union
                iou_scores.append(iou.item())
 
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    print(f"Mean IoU for test set: {mean_iou:.4f}")
    return mean_iou
 
test_dataset = SegmentationDataset(
    os.path.join(DATASET_PATH, "test/images"),
    os.path.join(DATASET_PATH, "test/masks"),
    processor=image_processor
)
 
mean_iou = calculate_mean_iou(model, test_dataset, DEVICE)
