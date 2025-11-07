import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import timm
from tqdm import tqdm

# Model Definition

class tbModel(nn.Module):
    def __init__(self, num_classes=2):
        super(tbModel, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




# Training Function

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Scheduler step
        scheduler.step(val_loss)

        print(f"📊 Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"💾 New best model saved with Val Acc: {val_acc:.4f}")

    print("Training complete. Best Val Accuracy:", best_val_acc)



# Main Script

if __name__ == "__main__":
    # Paths
    train_folder = './data/tb/train'
    valid_folder = './data/tb/valid'
    save_path = os.path.join(os.getcwd(), "tuberculosis_model.pth")

    # Hyperparameters
    num_classes = 2
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    img_size = 128  

    # Transforms
 class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return TF.pad(
            image,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=0 
        )

       

 transform = transforms.Compose([
    SquarePad(),                         # pad to make square
    transforms.Resize((128, 128)),       # resize to training size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
 ])

    # Datasets & Loaders
    train_dataset = ImageFolder(train_folder, transform=transform)
    val_dataset = ImageFolder(valid_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = tbModel(num_classes=num_classes).to(device)

    # Load checkpoint if exists
    if os.path.exists(save_path):
        print("🔄 Found existing model, loading for fine-tuning...")
        # model.load_state_dict(torch.load(save_path, map_location=device))
        state_dict = torch.load(save_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
          model.load_state_dict(state_dict["model_state_dict"])
        else:
          model.load_state_dict(state_dict)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path)
