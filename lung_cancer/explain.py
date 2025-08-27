import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LightweightModel  # Import your model
import torchvision.transforms as transforms
from PIL import Image

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = LightweightModel(input_size=384).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize for consistency
    transforms.ToTensor()
])

image_path = r"C:\Users\nanaq\OneDrive\Desktop\Work\Machine Learning\Deep Learning\Pytorch\recap\lung_cancer\images\mal1 (1).jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Function to predict for SHAP
def model_predict(x):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()

# Generate SHAP Explainer
explainer = shap.GradientExplainer(model_predict, input_tensor)
shap_values = explainer.shap_values(input_tensor)

# Visualize SHAP results
shap.image_plot(shap_values, input_tensor.cpu().numpy())
