import torchvision
import torchvision.transforms as transforms

def load_dataset(dataset_name):

    if dataset_name.lower() == 'mnist':
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
    
    elif dataset_name.lower() in ('cifar','cifar10','cifar-10'):

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    return full_train_dataset, test_dataset