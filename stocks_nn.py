import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms


#Preparing the dataset class
class TickerDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, label_paths, transform=None):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data_paths[index]
        ys = self.label_paths[index]
        return x, ys

    def __len__(self):
        return len(self.data_paths)

#Net model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, 3)
        self.conv2 = nn.Conv2d(50, 50, 3)
        self.fc1 = nn.Linear(500, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 50)
        self.fc4 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 500) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#Feedback on system performance
def net_check(net, testloader):
    df = pd.DataFrame(columns=['Predict Sell','Predict Hold','Predict Buy','Total/Overall'],
                          index=['Actual Sell','Actual Hold','Actual Buy', 'Total Predictions', 'Correct Predictions'])
    x=torch.empty(0)
    y=torch.empty(0)
    for i, data in enumerate(testloader, 0): 
        inputs, labels = data
        outputs = net(inputs)
        x = torch.cat((x,labels.float()),0)
        y = torch.cat((y,outputs),0)

    y = torch.argmax(y, dim=1)

    count = torch.zeros(5,4)
    for i in range(y.size()[0]):
        count[x[i].int()][y[i].int()]=count[x[i].int()][y[i].int()]+1

    for i in range(3):
        count[3,i] = count[:,i].sum()
        count[4,i] = count[i,i] / count[3,i]
    for i in range(4):
        count[i,3] = count[i].sum()
    count[4,3] = (count[0,0] + count[1,1] + count[2,2]) / count[3,3]

    df.loc['Actual Sell'] = pd.Series({'Predict Sell':int(count[0,0].item()), 'Predict Hold': int(count[0,1].item()),
                                           'Predict Buy': int(count[0,2].item()), 'Total/Overall': int(count[0,3].item())})
    df.loc['Actual Hold'] = pd.Series({'Predict Sell': int(count[1,0].item()), 'Predict Hold': int(count[1,1].item()),
                                           'Predict Buy': int(count[1,2].item()), 'Total/Overall': int(count[1,3].item())})
    df.loc['Actual Buy'] = pd.Series({'Predict Sell': int(count[2,0].item()), 'Predict Hold': int(count[2,1].item()),
                                           'Predict Buy': int(count[2,2].item()), 'Total/Overall': int(count[2,3].item())})
    df.loc['Total Predictions'] = pd.Series({'Predict Sell': int(count[3,0].item()), 'Predict Hold': int(count[3,1].item()),
                                           'Predict Buy': int(count[3,2].item()), 'Total/Overall': int(count[3,3].item())})
    df.loc['Correct Predictions'] = pd.Series({'Predict Sell': count[4,0].item(), 'Predict Hold': count[4,1].item(),
                                           'Predict Buy': count[4,2].item(), 'Total/Overall': count[4,3].item()})
    print(df)


#Parameters for TickerDataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((1585, 1585, 1585), (605, 605, 605))])
data_train = torch.load('stocks_nn/stock_data.pt')[:-(40+14)]
label_train = torch.load('stocks_nn/stock_labels.pt')[:-40]
data_test = torch.load('stocks_nn/stock_data.pt')[-(40+14):-14]
label_test = torch.load('stocks_nn/stock_labels.pt')[-40:]
data_predict = torch.load('stocks_nn/stock_data.pt')[-14:]

#Initialise dataset
trainset = TickerDataset(data_train, label_train, transform=transform)
testset = TickerDataset(data_test, label_test, transform=transform)

#Initialise net
net = Net()

#Training functions and data
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=0)

#Map for result classes
#classes = ['sell', 'hold', 'buy']

#Training process
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
print('Finished Training')

net_check(net, testloader) #using trainloader to test dataset
print("Prediction values: ", end='')
for value in torch.argmax(net(data_predict), dim=1):
    if value == 0:
        print('Sell, ', end='')
    if value == 1:
        print('Hold, ', end='')
    if value == 2:
        print('Buy, ', end='')
        
torch.save(net,'stocks_nn/stock_nn.pt')
