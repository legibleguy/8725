import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import PIL

class CIFAR10WithNoise(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_level=0.05):
        super().__init__(root, train, transform, target_transform, download)
        self.noise_level = noise_level

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Convert image to numpy array
        img_array = np.array(img)

        # Add salt and pepper noise
        noisy_img = self.add_salt_pepper_noise(img_array)

        # switch to Tensor
        noisy_img = np.transpose(noisy_img, (1, 2, 0))

        # Apply any additional transformations
        if self.transform is not None:
            noisy_img = self.transform(noisy_img)

        return noisy_img, target

    def add_salt_pepper_noise(self, img):
        # Amount of salt and pepper noise
        s_vs_p = 0.5
        amount = self.noise_level
        img = img.flatten()
        array_range = np.arange(img.shape[0])
        np.random.shuffle(array_range)

        # Add Salt noise
        num_salt = np.ceil(amount * img.size * s_vs_p)
        img[array_range[:int(num_salt)]] = 1

        # Add Pepper noise
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        if num_pepper > 0:
            img[array_range[-int(num_pepper):]] = 0
        img = img.reshape((3, 32, 32))
        return img
    
class CIFAR10_DataLoader():
    def __init__(self, batch_size, noise_level, test_noise=0):
        transform = transforms.Compose([transforms.ToTensor()])
        
        test_noise = noise_level * test_noise
        train_dataset = CIFAR10WithNoise(root='./data', train=True, download=True, transform=transform, noise_level=noise_level)
        test_dataset = CIFAR10WithNoise(root='./data', train=False, download=True, transform=transform, noise_level=test_noise)
        
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_classes(self):
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def get_num_classes(self):
        return len(self.get_classes())
    
    def get_input_shape(self):
        return (3, 32, 32)
    
    def get_output_shape(self):
        return (self.get_num_classes(),)
    
    def get_train_size(self):
        return len(self.train_loader.dataset)
    
    def get_test_size(self):
        return len(self.test_loader.dataset)
    
    def get_train_batches(self):
        return len(self.train_loader)
    
    def get_test_batches(self):
        return len(self.test_loader)
    
    def get_train_data(self):
        return self.train_loader.dataset.data
    
    def get_test_data(self):
        return self.test_loader.dataset.data
    
    def get_train_labels(self):
        return self.train_loader.dataset.targets
    
    def get_test_labels(self):
        return self.test_loader.dataset.targets

if __name__ == "__main__":
    dataloaders = CIFAR10_DataLoader(128, 0.00)
    train_loader = dataloaders.get_train_loader()
    test_loader = dataloaders.get_test_loader()
    print(len(train_loader.dataset))
    print(len(test_loader.dataset))
    for images, labels in train_loader:
        image = images[0].numpy().transpose(1, 2, 0)
        # save np array as png file
        img = PIL.Image.fromarray((image * 255).astype(np.uint8))
        img.save('my.png')
        print(images.shape)
        print(labels.shape)
        break
    for images, labels in test_loader:
        print(images.shape)
        print(labels.shape)
        break
