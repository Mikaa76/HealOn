import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import timm
import os



# Model Definition

class tbModel(nn.Module):
    """Custom TB classification model based on EfficientNet-B0"""

    def __init__(self, num_classes=2):
        super(tbModel, self).__init__()
        # Load EfficientNet-B0 
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Remove classifier head, keep only feature extractor
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # EfficientNet-B0 thing
        enet_out_size = 1280

        # classifier for binary classification (Normal vs TB)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


#checking if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path
base_dir = os.path.dirname(os.path.dirname(__file__))  
model = os.path.join(base_dir, "models", "tuberculosis_model.pth")
MODEL_PATH = model

# Initialize model
model = tbModel(num_classes=2).to(device)

# Loading trained model
state_dict = torch.load(MODEL_PATH, map_location=device)
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    model.load_state_dict(state_dict)

model.eval()  # set to evaluation mode



# Image Preprocessing
# Actually we need to resize image to same size on which our model is trained on..
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),   # resize to training size
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],  # ImageNet normalization
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# resize image to square if its not a square image
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

# Class labels from training
classes = ["Normal", "Tuberculosis"]



# Prediction Function
def predict_image(image_path):
    """
    Run inference to checkf or tb

    It will return label and confidence 
    """
    # Load image 
    img = Image.open(image_path).convert("RGB")
    
    # Apply transforms and add batch dimension
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[pred_idx].item()
    # we will get output as 0 or 1 so we can grab that value from our classes
    return classes[pred_idx], confidence



# testing zone ahead
if __name__ == "__main__":

    img_path = "tb.jpeg"

    label, conf = predict_image(img_path)
    print(f"{img_path} → {label} (confidence: {conf:.3f})")
