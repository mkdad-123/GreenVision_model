import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F


num_classes = 38  

def load_model(weights_path="best_model18.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    weights = torch.load(weights_path , map_location="cpu")
    model.load_state_dict(weights['model_state_dict'])
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def predict_image(image: Image.Image, model, class_names, topk=3):
    img_tensor = transform(image).unsqueeze(0)  # batch size = 1
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)  
        top_probs, top_idxs = probs.topk(topk, dim=1)  
        
    results = []
    for i in range(topk):
        results.append({
            "class": class_names[top_idxs[0][i].item()],
            "probability": round(100 * float(top_probs[0][i].item()), 2),
            "id" : top_idxs[0][i].item(),
        })
    
    return results

