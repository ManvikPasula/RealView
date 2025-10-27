# implementation for https://github.com/xiaoyufenfei/LEDNet

from google.colab import drive
drive.mount('/content/drive')
 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate as interpolate
from PIL import Image
 
DATASET_PATH = "/content/drive/MyDrive/Research/RealView/seg_dataset"
MODEL_SAVE_PATH = "/content/drive/MyDrive/Research/RealView/model_paths/"
CHECKPOINT_PATH = None  
BATCH_SIZE = 16  
NUM_WORKERS = 12
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
IMG_SIZE = (640, 480)  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  
CHECKPOINT_INTERVAL = 5  
 
class SegmentationDataset(Dataset):
  def __init__(self, image_dir, mask_dir, transform=None):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.image_files = sorted(os.listdir(image_dir))
    self.mask_files = sorted(os.listdir(mask_dir))
    self.transform = transform
 
  def __len__(self):
    return len(self.image_files)
 
  def __getitem__(self, idx):
    img_path = os.path.join(self.image_dir, self.image_files[idx])
    mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
 
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
 
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = mask // 255
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  
    mask = torch.tensor(mask, dtype=torch.long) 
 
    return image, mask
 
train_dataset = SegmentationDataset(
    os.path.join(DATASET_PATH, "train/images"), os.path.join(DATASET_PATH, "train/masks")
)
valid_dataset = SegmentationDataset(
    os.path.join(DATASET_PATH, "valid/images"), os.path.join(DATASET_PATH, "valid/masks")
)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate
 
def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2
 
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x
 
class Conv2dBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.conv(x)
 
class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super(SS_nbt_module,self).__init__()
 
        oup_inc = chann//2
 
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
 
        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
 
        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
 
        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
 
        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))
 
        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
 
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
 
        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
 
        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)
 
        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
 
        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))
 
        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)
 
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)
 
    @staticmethod
    def _concat(x,out):
        return torch.cat((x,out),1)
 
    def forward(self, input):
 
 
        residual = input
        x1, x2 = split(input)
 
        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)
 
        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
 
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)
 
        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)
 
        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)
 
        out = self._concat(output1,output2)
        out = F.relu(residual + out, inplace=True)
 
        return channel_shuffle(out, 2)
 
class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output
 
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 32)
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))
        self.layers.append(DownsamplerBlock(32, 64))
        for _ in range(2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for _ in range(1):
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
        for _ in range(1):
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            self.layers.append(SS_nbt_module(128, 0.3, 17))
 
    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        return output
 
class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dBnRelu(in_ch, out_ch, kernel_size=1))
        self.mid = nn.Sequential(Conv2dBnRelu(in_ch, out_ch, kernel_size=1))
        self.down1 = Conv2dBnRelu(in_ch, 128, kernel_size=7, stride=2, padding=3)
        self.down2 = Conv2dBnRelu(128, 128, kernel_size=5, stride=2, padding=2)
        self.down3 = nn.Sequential(
            Conv2dBnRelu(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(128, 1, kernel_size=1)          )
        self.conv2 = Conv2dBnRelu(128, 1, kernel_size=1)  
        self.conv1 = Conv2dBnRelu(128, 1, kernel_size=1)  
 
    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        b1 = interpolate(self.branch1(x), size=(h, w), mode="bilinear", align_corners=True)
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = interpolate(self.down3(x2), size=(h // 4, w // 4), mode="bilinear", align_corners=True)
        x2 = self.conv2(x2)
        x = x2 + x3
        x = interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        x1 = self.conv1(x1)
        x = x + x1
        x = interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        x = torch.mul(x, mid)
        x = x + b1
        return x
 
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.apn = APN_Module(in_ch=128, out_ch=1) 
 
    def forward(self, input):
        output = self.apn(input)
        return interpolate(output, size=(480, 640), mode="bilinear", align_corners=True)
 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
 
    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return torch.argmax(torch.sigmoid(output), dim=1)
 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
 
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  
 
        if inputs.shape != targets.shape:
            targets = F.one_hot(targets.long(), num_classes=inputs.shape[1])  
            targets = targets.permute(0, 3, 1, 2)  
 
        inputs = inputs.contiguous()
        targets = targets.contiguous()
 
        intersection = (inputs * targets).sum(dim=(2, 3))  
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
 
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  
 
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
 
    def forward(self, inputs, targets):
        if inputs.shape[1] > 1:  
            targets = F.one_hot(targets.long(), num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2)  
 
        targets = targets.float()
 
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
 
        return focal_loss.mean()
 
 
 
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
 
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal
 
model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
cudnn.benchmark = True
 
criterion = CombinedLoss(dice_weight=0.7, focal_weight=0.3)
 
def train_model(patience=5, alpha=0.5): 
    start_epoch = 0
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
 
    dice_criterion = DiceLoss()
    focal_criterion = FocalLoss(alpha=0.25, gamma=2)
 
    if CHECKPOINT_PATH is not None and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"Resumed training from epoch {start_epoch}")
 
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(DEVICE):
                outputs = model(images)
                dice_loss = dice_criterion(outputs, masks)
                focal_loss = focal_criterion(outputs, masks)
                loss = alpha * dice_loss + (1 - alpha) * focal_loss  
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss_history.append(loss.item()) 
 
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_loss:.4f}")
 
        print("Doing validation")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(valid_loader):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                dice_loss = dice_criterion(outputs, masks)
                focal_loss = focal_criterion(outputs, masks)
                loss = alpha * dice_loss + (1 - alpha) * focal_loss
                val_loss += loss.item()
                val_loss_history.append(loss.item()) 
 
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
 
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_filename = os.path.join(MODEL_SAVE_PATH, f"lednet_checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }, checkpoint_filename)
            print(f"Checkpoint saved at epoch {epoch+1}")
 
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "lednet_colab.pth"))
    print("Final model saved!")
 
    return train_loss_history, val_loss_history
 
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
 
from sklearn.metrics import jaccard_score
import torchmetrics
import torch
import numpy as np
 
def load_model(model_path, num_classes=2, device="cuda"):
    model = Net(num_classes)  
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model
 
def preprocess_image(image_path, input_size=(640, 480)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size)
 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  
    return image, image_tensor
 
def preprocess_mask(mask_path, input_size=(640, 480)):
    mask = Image.open(mask_path).convert("L")  
    mask = mask.resize(input_size)
    mask = np.array(mask)
 
    mask[mask > 0] = 1
 
    return torch.tensor(mask, dtype=torch.long)
 
 
def evaluate_model(model, test_images_path, test_masks_path, criterion, device="cuda"):
    total_iou = 0
    total_dice = 0
    total_loss = 0
    num_samples = 0
 
    print("Testing model")
 
    for image_filename in tqdm(os.listdir(test_images_path)):
        image_path = os.path.join(test_images_path, image_filename)
        mask_path = os.path.join(test_masks_path, image_filename.replace(".png", "_mask.png"))
 
        if not os.path.exists(mask_path):  
            continue
 
        image, image_tensor = preprocess_image(image_path)
        mask_tensor = preprocess_mask(mask_path)  
        image_tensor = image_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
 
        with torch.no_grad():
            output = model(image_tensor)  
            output = torch.sigmoid(output)  
            pred_mask = (output > 0.5).float().squeeze(0).cpu().numpy()  
 
        mask_tensor = mask_tensor.cpu().numpy()
        pred_mask = pred_mask.squeeze(0)
 
        iou = jaccard_score(mask_tensor.flatten(), pred_mask.flatten(), average="binary")
 
        intersection = np.sum(pred_mask * mask_tensor)
        dice_score = (2. * intersection) / (np.sum(pred_mask) + np.sum(mask_tensor) + 1e-6)
 
        mask_tensor_torch = torch.tensor(mask_tensor, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)  
        loss = criterion(output, mask_tensor_torch)  
 
        total_iou += iou
        total_dice += dice_score
        total_loss += loss.item()
        num_samples += 1
 
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_loss = total_loss / num_samples
 
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice: {avg_dice:.4f}")
    print(f"Loss: {avg_loss:.4f}")
    return avg_iou, avg_dice, avg_loss
 
 
 
MODEL_PATH = "/content/drive/MyDrive/Research/RealView/model_paths/lednet_checkpoint_epoch50.pth"
TEST_IMAGES_PATH = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/images"
TEST_MASKS_PATH = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/masks"
 
model = load_model(MODEL_PATH)
evaluate_model(model, TEST_IMAGES_PATH, TEST_MASKS_PATH)
 
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
 
def load_model(model_path, num_classes=2, device="cuda"):
    model = Net(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model
 
def preprocess_image(image_path, input_size=(640, 480)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size)
 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  
    return image, image_tensor
 
def load_ground_truth(mask_path, input_size=(640, 480)):
    mask = Image.open(mask_path).convert("L")  
    mask = mask.resize(input_size)
    mask = np.array(mask)  
    return mask
 
def infer_and_display(model, image_path, mask_path=None, device="cuda"):
    image, image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
 
    actual_mask = None
    if mask_path is not None:
        actual_mask = load_ground_truth(mask_path)
 
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
 
    predicted_mask_display = (predicted_mask * 255).astype(np.uint8)
 
    if actual_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
 
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
 
    if actual_mask is not None:
        axes[1].imshow(actual_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        pred_idx = 2
    else:
        pred_idx = 1  
 
    axes[pred_idx].imshow(predicted_mask_display, cmap="gray")
    axes[pred_idx].set_title("Predicted Mask")
    axes[pred_idx].axis("off")
 
    plt.show()
 
MODEL_PATH = "/content/drive/MyDrive/Research/RealView/model_paths/lednet_checkpoint_epoch50.pth"
model = load_model(MODEL_PATH)
 
IMAGE_PATH = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/images/2_walkway_test_9.png"
MASK_PATH = "/content/drive/MyDrive/Research/RealView/seg_dataset/test/masks/2_walkway_test_9_mask.png"  
 
infer_and_display(model, IMAGE_PATH, MASK_PATH)  
infer_and_display(model, IMAGE_PATH, None)
