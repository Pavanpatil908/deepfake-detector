import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")  
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)


model.load_state_dict(torch.load("deepfake_model.pth", map_location=device), strict=False)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
   
    img_tensor = transform(img).unsqueeze(0).to(device)

    
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)

    
    label = "Real" if prediction.item() == 0 else "Fake"
    color = (0, 255, 0) if label == "Real" else (0, 0, 255)

    
    cv2.putText(frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Deepfake Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


