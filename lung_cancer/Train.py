from torchvision import datasets, transforms
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchsummary import summary
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
from model import LightweightModel

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)


# display image function
def imshow(img, labels, title):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

    # print labels
    label_names = [classnames[label] for label in labels]
    print("Labels:", label_names)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    plt.savefig("Confusion Matrix.jpg")
    plt.show()


# Example usage
model = LightweightModel(input_size=384, lstm_units=64, tcn_channels=64, num_classes=3)
model.to(device)
# print(model)
print(summary(model, input_size=(1 ,3 ,128 ,128)))

# Loading data

# dataset loading
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_ratio = 0.6
val_ratio = 0.3

path = r"E:\Data\Augmented IQ-OTHNCCD lung cancer dataset"

dataset = datasets.ImageFolder(root=path, transform=transform)

# spliting dataset
train_size = round(train_ratio * len(dataset))
val_size = round(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classnames = dataset.classes

# Checking dataset sizes
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f'{classnames}')

# Displaying image with labels
# get a batch on images
dataiter = iter(val_loader)
images, labels = next(dataiter)

# display images with labels
imshow(make_grid(images[:5]), labels[:5], title="Validation images")

# model initialization and training
learning_rate = 1e-4
num_epochs = 30

# defining loss function
criterion = nn.CrossEntropyLoss()
# optimizers
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  
#optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)


# Define the scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.15, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize lists to store training & validation metrics
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Initialize best validation loss to a high value
best_val_loss = float('inf')
best_model_path = "best_model.pth"  # File path for saving best model

n_total_steps = len(val_loader)
print(f"Total Validation Steps per Epoch: {n_total_steps}")

for epoch in range(num_epochs):
    ### TRAINING PHASE ###
    model.train()  # Set model to training mode
    running_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Track training loss & accuracy
        running_loss += loss.item() * inputs.size(0)  # Sum loss (scaled by batch size)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    # Calculate epoch losses and accuracies
    train_running_loss = running_loss / total
    train_running_accs = correct / total

    # Append to tracking lists
    train_losses.append(train_running_loss)
    train_accs.append(100. * train_running_accs)

    ### VALIDATION PHASE ###
    model.eval()  # Set model to evaluation mode
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():  # No gradients needed for validation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Track validation loss & accuracy
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    # Calculate epoch losses and accuracies
    val_running_loss = val_loss / total
    val_running_accs = correct / total

    # Append to tracking lists
    val_losses.append(val_running_loss)
    val_accs.append(100. * val_running_accs)

    # Step the scheduler (if using ReduceLROnPlateau)
    scheduler.step(val_running_loss)

    # Print progress every few epochs
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {train_running_loss:.4f}, Acc: {train_running_accs:.4f} | "
              f"Val Loss: {val_running_loss:.4f}, Acc: {val_running_accs:.4f}")

    # **Save Best Model Checkpoint**
    if val_running_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} → {val_running_loss:.4f}. Saving model...")
        best_val_loss = val_running_loss
        torch.save(model.state_dict(), best_model_path)

print("Training Complete. Best model saved as 'best_model.pth'.")

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# plt.grid()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training & Validation Accuracy')
plt.legend()
# plt.grid()
plt.savefig("Curves.jpg")

plt.show()

# Evaluation with the test set
model.load_state_dict(torch.load(best_model_path))
model.to(device)  # Move to GPU if available


def evaluate(model, loader):
    y_true, y_pred, y_probs = [], [], []
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collecting true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=dataset.classes)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #Compute AUC
    y_probs = np.array(y_probs)  # Convert to NumPy array for sklearn compatibility

    Auc = roc_auc_score(y_true, y_probs, multi_class="ovr")  # Use positive class probabilities

    return accuracy, report, cm, Auc


# Call the evaluate function
test_accuracy, test_classification_report, test_confusion_matrix,auc = evaluate(model, test_loader)

# Print results
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Auc: {auc:.2f}%')
print(test_classification_report)

# Plot the confusion matrix
plot_confusion_matrix(test_confusion_matrix, dataset.classes)

