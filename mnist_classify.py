import numpy
import random
import torch
import torchvision
import matplotlib.pyplot as plt
import os
def plotDigits(digitInfo):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    for i in range(0, 10):
        plt.imshow(numpy.asarray(digitInfo[0][i]).reshape((28, 28)))
        plt.show()
        print("Predicted Label: ", str(digitInfo[1][i]), "Actual Label: ", str(digitInfo[2][i]))

def calcTestAccuracy(model, test_load):
    correct = 0
    total = 0
    for images, labels in test_load:
        with torch.no_grad():
            image = images.view(-1, 784)
            outputs = model(image)
            maxes, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct/total
    return accuracy

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def net1(mnist_train, mnist_test, train_load, test_load, valid_load):
    epochs = 2
    epochCount = 0
    batch_size = 100
    n_iters = 3000
    input_dim = 784
    output_dim = 10
    model = LogisticRegression(input_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    iteration = 0
    select = False
    choose = random.randint(1, epochs)
    prevLoss = 10000000
    for epoch in range(0, epochs):
        print("EPCOH: ", epoch)
        for i, (images, labels) in enumerate(train_load):
            images = torch.Tensor(images.view(images.shape[0], -1))
            labels = labels.clone().detach()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)#computes softmax + entropy loss
            loss.backward()
            optimizer.step()
            iteration +=1
            if (iteration % len(train_load) == 0):
                epochCount +=1
                if(epochCount == choose):
                    select = True
                correct = 0
                total = 0
                for images, labels in valid_load:
                    with torch.no_grad():
                        image = images.view(-1, 784)
                        outputs = model(image)
                        maxes, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        if (select == True):
                            imagesToPlot = (images.numpy()[0:10], predicted.numpy()[0:10], labels.numpy()[0:10])
                            select = False
                if epochCount == 1:
                    prevLoss = loss.item()
                    torch.save({'model': model,'loss': loss}, "/Users/mnadig/Desktop/best_model.pt")
                else:
                    if loss.item() < prevLoss:
                        torch.save({'model': model,'loss': loss}, "/Users/mnadig/Desktop/best_model.pt")
                        prevLoss = loss.item()
                accuracy = 100 * correct/total
                print("ITERATION: ", iteration, "EPOCH: ", epochCount, "LOSS: ", loss.item(), "ACCURACY: ", accuracy)
    #plotDigits(imagesToPlot)
    m = torch.load("/Users/mnadig/Desktop/best_model.pt")
    return m

class net2(torch.nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.hidden = torch.nn.Linear(784, 512)
        self.output = torch.nn.Linear(512, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

def net2_mnist(mnist_train, mnist_test, train_load, test_load, valid_load):
    epochs = 2
    epochCount = 0
    batch_size = 100
    model = net2()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    iteration = 0
    for epoch in range(0, epochs):
        for i, (images, labels) in enumerate(train_load):
            images = torch.Tensor(images.view(images.shape[0], -1))
            labels = labels.clone().detach()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)#computes softmax + entropy loss
            loss.backward()
            optimizer.step()
            iteration +=1
            if (iteration % len(train_load) == 0):
                epochCount +=1
                correct = 0
                total = 0
                for images, labels in valid_load:
                    with torch.no_grad():
                        image = images.view(-1, 784)
                        outputs = model(image)
                        maxes, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                if epochCount == 1:
                    prevLoss = loss.item()
                    torch.save({'model': model,'loss': loss}, "/Users/mnadig/Desktop/best_model2.pt")
                else:
                    if loss.item() < prevLoss:
                        torch.save({'model': model,'loss': loss}, "/Users/mnadig/Desktop/best_model2.pt")
                        prevLoss = loss.item()
                accuracy = 100 * correct/total
                print("ITERATION: ", iteration, "EPOCH: ", epochCount, "LOSS: ", loss.item(), "ACCURACY: ", accuracy)
    #plotDigits(imagesToPlot)
    m = torch.load("/Users/mnadig/Desktop/best_model2.pt")
    return m

class net3(torch.nn.Module):
    def __init__(self):
        super(net3, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 1,out_channels=32, kernel_size=5, stride=1, padding=2, padding_mode = 'zeros')
        self.mp1 = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels = 64, kernel_size=5, stride=1, padding=2, padding_mode = 'zeros')
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(3136, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.fc1(x.view(-1, 7*7*64))
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def net3_mnist(mnist_train, mnist_test, train_load, test_load, valid_load):
    epochs = 100
    epochCount = 0
    batch_size = 50
    model = net3()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    iteration = 0
    for epoch in range(0, epochs):
        for i, (images, labels) in enumerate(train_load):
            labels = labels.clone().detach()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)#computes softmax + entropy loss
            loss.backward()
            optimizer.step()
            iteration +=1
            if (iteration % len(train_load) == 0):
                epochCount +=1
                correct = 0
                total = 0
                for images, labels in valid_load:
                    with torch.no_grad():
                        outputs = model(images)
                        maxes, predicted = torch.max(outputs.data, 1)
                        for i in range(0, len(labels.numpy())):
                            if(predicted.numpy()[i] == labels.numpy()[i]):
                                correct +=1
                        total +=  50
                if epochCount == 1:
                    prevLoss = loss.item()
                    torch.save({'model': model,'loss': loss}, "/Users/mnadig/Desktop/best_model3.pt")
                else:
                    if loss.item() < prevLoss:
                        torch.save({'model': model,'loss': loss}, "/Users/mnadig/Desktop/best_model3.pt")
                        prevLoss = loss.item()
                accuracy = 100 * correct/total
                print("ITERATION: ", iteration, "EPOCH: ", epochCount, "LOSS: ", loss.item(), "ACCURACY: ", accuracy)
    #plotDigits(imagesToPlot)
    m = torch.load("/Users/mnadig/Desktop/best_model3.pt")
    return m
    
def best_model_acc(model1, model2, model3, test_load):
    models = [model1, model2, model3]
    losses = [model1['loss'], model2['loss'], model3['loss']]
    best_model = models[losses.index(min(losses))]['model']
    acc = calcTestAccuracy(best_model, test_load)
    print(acc)
    return acc

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST("/Users/mnadig/Desktop", train = True, download = True, transform=transform)
    dataset = torchvision.datasets.MNIST("/Users/mnadig/Desktop", train = False, download = True, transform=transform)
    
    mnist_test, mnist_valid = torch.utils.data.random_split(dataset, [5000,5000])
    train_load = torch.utils.data.DataLoader(mnist_train, shuffle = True, batch_size = 50)
    valid_load = torch.utils.data.DataLoader(mnist_valid, shuffle = True, batch_size = 50)
    test_load = torch.utils.data.DataLoader(mnist_test, shuffle = False, batch_size = 50)
    
    #model1 = net1(mnist_train, mnist_test, train_load, test_load, valid_load)   
    #model2 = net2_mnist(mnist_train, mnist_test, train_load, test_load, valid_load)
    model3 = net3_mnist(mnist_train, mnist_test, train_load, test_load, valid_load)                       
    
    #test_accuracy= best_model_acc(model1, model2, model3, test_load)   
