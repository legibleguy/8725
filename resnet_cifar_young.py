import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.models import resnet18
import argparse
from CIFAR10_DataLoader import CIFAR10_DataLoader
import sys

# Define the arguments
parser = argparse.ArgumentParser(description='Train a ResNet on CIFAR10')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--noise_level', type=float, default=0.05, help='Amount of noise to add to CIFAR10')
parser.add_argument('--test_noise', type=int, default=0, help='Should the test set have noise? 1 for yes, 0 for no')
parser.add_argument('--filename', type=str, default='experiment.txt', help='Filename to save the trained model')
args = parser.parse_args()
true_stdout = sys.stdout
with open(args.filename, 'w') as f:
    sys.stdout = f

    # Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dl_class = CIFAR10_DataLoader(batch_size=args.batch_size, noise_level=args.noise_level, test_noise=args.test_noise)
    trainloader = dl_class.get_train_loader()
    testloader = dl_class.get_test_loader()

    # Define a ResNet model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)  # CIFAR10 has 10 classes
    model = model.to(device)  # Move model to GPU if available

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    print(args.lr)

    # Train the model
    for epoch in range(args.epochs):
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # Move inputs and labels to GPU if available
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_running_loss += loss.item()
        
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_running_loss += loss.item()

        print('[%d] Train loss: %.3f Test loss: %.3f Train Accuracy: %.3f Test Accuracy: %.3f' %
                (epoch + 1, train_running_loss / 50000, test_running_loss / 10000, 100 * train_correct / train_total, 100 * test_correct / test_total))
        sys.stdout = true_stdout
        print('[%d] Train loss: %.3f Test loss: %.3f Train Accuracy: %.3f Test Accuracy: %.3f' %
                (epoch + 1, train_running_loss / 50000, test_running_loss / 10000, 100 * train_correct / train_total, 100 * test_correct / test_total))
        sys.stdout = f
    print('Finished Training')

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # Move images and labels to GPU if available
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))
