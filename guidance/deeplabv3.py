# implementation for https://github.com/leimao/DeepLab-V3

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
 
DATASET_PATH = "/content/drive/MyDrive/Research/RealView/seg_dataset"
MODEL_SAVE_PATH = "/content/drive/MyDrive/Research/RealView/deeplabv3/model_paths/"
CHECKPOINT_PATH = "/content/drive/MyDrive/Research/RealView/deeplabv3/model_paths/deeplabv3_checkpoint_epoch15.pth"  
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 1e-5
IMG_SIZE = (640, 480)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_INTERVAL = 5
 
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
 
    def __len__(self):
        return len(self.image_filenames)
 
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace(".png", "_mask.png"))
 
        if mask_path.endswith("(1)_mask.png"):
          print(img_path)
          print(mask_path)
 
       
        image = cv2.imread(img_path)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
 
       
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
 
       
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  
        try:
          mask = torch.tensor(mask, dtype=torch.float32) / 255.0  
        except:
          print(img_path)
          print(mask_path)
          print(mask)
          print(image)
 
        return image, mask
 
train_dataset = SegmentationDataset(
    os.path.join(DATASET_PATH, "train/images"), os.path.join(DATASET_PATH, "train/masks")
)
valid_dataset = SegmentationDataset(
    os.path.join(DATASET_PATH, "valid/images"), os.path.join(DATASET_PATH, "valid/masks")
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
        inputs = torch.sigmoid(inputs)  
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return self.alpha * self.bce(inputs, targets) + (1-self.alpha) * (1 - dice)  
 
model = models.deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  
model.to(DEVICE)
 
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
            images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss_history.append(loss.item())
 
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_loss:.4f}")
 
       
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(valid_loader):
                images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.float32).unsqueeze(1)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_loss_history.append(loss.item())
 
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(" Early stopping triggered!")
                break
 
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"deeplabv3_checkpoint_epoch{epoch+1}.pth"))
 
    return train_loss_history, val_loss_history
 
torch.cuda.empty_cache()
 
train_history, val_history = train_model(5)
 
def plot_loss(train_loss_history, val_loss_history):
    plt.figure(figsize=(10, 5))
 
   
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_history, label="Training Loss", color='blue')
    plt.xlabel("Batch Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
 
   
    plt.subplot(2, 1, 2)
    plt.plot(val_loss_history, label="Validation Loss", color='red')
    plt.xlabel("Batch Steps")
    plt.ylabel("Loss")
    plt.title("Validation Loss Over Time")
    plt.legend()
    plt.grid(True)
 
    plt.tight_layout()
    plt.show()
 
plot_loss(train_history, val_history)
 
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
 
def infer_and_display(model, image_path, mask_path=None, device="cuda"):
   
    image = Image.open(image_path).convert("RGB").resize((640, 480))
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
 
   
    with torch.no_grad():
        output = model(image_tensor)['out']  
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()  
        pred_mask = (pred_mask > 0.5).astype(np.uint8)  
 
   
    fig, axes = plt.subplots(1, 3 if mask_path else 2, figsize=(15, 5))
 
   
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
 
   
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L").resize((640, 480))
        mask = np.array(mask)
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
 
       
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")
    else:
       
        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")
 
    plt.show()
 
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "deeplabv3_checkpoint_epoch20.pth")
 
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
 
test_image_path = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/images/2_netra_test_9.png"
test_mask_path = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/masks/2_netra_test_9_mask.png"
 
infer_and_display(model, test_image_path, test_mask_path, DEVICE)
