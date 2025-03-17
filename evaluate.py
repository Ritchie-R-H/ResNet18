import torch
from model import CustomResNet
from sklearn.metrics import confusion_matrix, classification_report
from dataset import load_data
import numpy as np
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

def evaluate_model():
    # load data
    _, testloader = load_data()
    
    # load model
    model = CustomResNet()
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Grad-CAM
    #  cam_extractor = GradCAM(model, target_layer='layer4')

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    # compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # compute classification report
    report = classification_report(all_labels, all_predictions)
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    evaluate_model()
