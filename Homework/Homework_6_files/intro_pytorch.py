import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)
        
    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    #dataset transformer
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST("./data",train=True, download=True,transform=custom_transform)
    
    test_set=datasets.FashionMNIST("./data", train=False, transform=custom_transform)

    if training == True: 
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)

    return loader

def build_model():
    """
    TODO: implement this function.
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(), 
        nn.Linear(128, 64),
        nn.Linear(64, 10))

    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training
    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epochs in range(T):
        model.train()
        correct = 0
        total = 0
        loss_output = 0 
        for batch_idx, (data, target) in enumerate(train_loader):
            opt.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            opt.step()

            loss_output += loss.item()
            predicted = outputs.argmax(dim=1, keepdim=True)
            total += data.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

        avg_loss = loss_output/total 
        accuracy = (correct/total)*100
        print(f'Train Epoch: {epochs} Accuracy: {correct}/{total} ({accuracy:.2f}%) Loss: {avg_loss:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    correct = 0
    total = 0
    loss_output = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)

            loss_output += loss.item()
            predicted = outputs.argmax(dim=1, keepdim=True)
            total += data.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

        avg_loss = loss_output/total 
        accuracy = correct/total*100

    if show_loss:
        print(f'Average Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')



def predict_label(model, test_images, index):
    """
    TODO: implement this function.
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1
    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        output = model(test_images[index].unsqueeze(0))
        prob = F.softmax(output, dim=1)
        top3_prob, top3_classes = torch.topk(prob, 3, dim = 1)  
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        for i in range(3):
            class_idx = top3_classes[0][i].item()
            class_name = class_names[class_idx]
            class_prob = top3_prob[0][i].item() * 100  
            print(f'{class_name}: {class_prob:.2f}%')



if __name__ == '__main__':
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)

    model = build_model()
    print(model)

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = False)
    evaluate_model(model, test_loader, criterion, show_loss = True)

    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 1)