import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import pickle
import os
from PIL import Image
from model import CustomResNet


from matplotlib import pyplot as plt

# Custom CIFAR10 dataset
class CIFAR10Dataset(Dataset):
    def __init__(self, data, ids, transform=None):
        self.data = data # (N, 3, 32, 32)
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        ids = self.ids[idx]
        
        if self.transform:
            img = self.transform(img)
        return img, ids


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(batch_size = 128):
    data_dir = 'rh/data'
    
    #using custom cifar10 dataset
    def load_kaggle_test():
        test_file = os.path.join(data_dir, "cifar_test_nolabel.pkl")
        test_dict = unpickle(test_file)
        
        test_data = test_dict[b'data'].reshape((-1, 32, 32, 3)).astype(np.uint8)
        test_ids = test_dict[b'ids']
        
        return test_data, test_ids
    

    # Load Data
    test_data, test_ids = load_kaggle_test()


    # Calculate mean and std for normalization
    mean = np.mean(test_data / 255.0, axis=(0, 1, 2))
    std = np.std(test_data / 255.0, axis=(0, 1, 2))
    print(f'Mean: {mean}')
    print(f'Std: {std}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Create datasets
    test_dataset = CIFAR10Dataset(test_data, test_ids, transform=transform)
    batch_size = 128

    print(f"test_ids: {test_ids}")
    batch_size = 128

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return test_loader

def save_images_as_jpg(test_loader, pred_labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    
    for i, (img, _) in enumerate(test_loader.dataset):
        img = img.permute(1, 2, 0).numpy()  # convert to numpy array
        img = (img * std + mean) * 255.0  
        img = img.astype(np.uint8) # multiply by 255 and convert to uint8
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'image_{i}_{pred_labels[i]}.jpg'))

def show_image(img_tensor, label):
    # labels
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    img = img_tensor.permute(1, 2, 0).numpy()
    
    # adjust image to valid range [0, 1]
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    img = (img * std + mean) * 255.0
    img = img.astype(np.uint8)
    # show img
    plt.imshow(img)
    plt.title(f"Label: {classes[label]}")
    plt.axis('off')  # Hide axes for better image display
    plt.show()

if __name__ == "__main__":

    testloader = load_data()

    model = CustomResNet()
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, ids in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # show images in test set
    start_index = 0
    end_index = 0
    for i in range(start_index, end_index):
        show_image(testloader.dataset[i][0], predictions[i])

    # Save all test images as JPG
    #output_dir = 'kaggle_test_images'
    #save_images_as_jpg(testloader, predictions, output_dir)


    with open('submission.csv', 'w') as f:
        f.write('ID,Labels\n')
        for i, pred in enumerate(predictions):
            f.write(f'{i},{pred}\n')

    print("Submission file created")
