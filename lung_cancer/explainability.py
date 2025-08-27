from model import LightweightModel
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import cv2
from Grad_CAM import GradCAM


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)


#dataset
path = r"E:\Data\Augmented IQ-OTHNCCD lung cancer dataset"
dataset = datasets.ImageFolder(root=path)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
class_names = dataset.classes
print(class_names)


# model loading
model = LightweightModel(input_size= 384)
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)


def predict_fn(images):
    """
    Function that takes a list of images and returns the model's predicted probabilities.
    """
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Ensure resized image matches model expectation
        transforms.ToTensor()
    ])

    # Convert LIME perturbed images to tensors
    images = [transform(Image.fromarray(img.astype('uint8'))) for img in images]
    images = torch.stack(images).to(device)  # Stack images into a batch

    with torch.no_grad():
        outputs = model(images)  # Get raw logits
        probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities

    return probs.cpu().numpy()


# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust based on model input size
    transforms.ToTensor()
])


image_path = r"C:\Users\nanaq\OneDrive\Desktop\Work\Machine Learning\Deep Learning\Pytorch\recap\lung_cancer\images\ben (3).jpg"  # Path to test image
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
actual_class = "Benign"


explainer = lime_image.LimeImageExplainer()

# Explain the model's prediction for the given image
explanation = explainer.explain_instance(
    np.array(image),  # Convert PIL image to NumPy array
    predict_fn,       # Model prediction function
    top_labels=5,     # Number of top labels to explain
    hide_color=0,     # Hide background color
    num_samples=1000  # Number of perturbed samples
)

# Get explanation for the most probable class
top_class = explanation.top_labels[0]
temp, mask = explanation.get_image_and_mask(
    label=top_class, positive_only=True, hide_rest=True, num_features=5
)

# Show LIME explanation
plt.imshow(mark_boundaries(temp, mask))
plt.title(f"Actual: {actual_class} | Predicted: {class_names[top_class]}")
#plt.title(f"LIME Explanation for Class {top_class}")
plt.axis("off")
#plt.savefig("Ben_3.png")
plt.show()

