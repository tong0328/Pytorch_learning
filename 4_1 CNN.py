import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from matplotlib import pyplot as plt
import numpy as np


#Hyper Parameters
EPOCH = 3
BATCH_SIZE = 50                  #60000/50=1200
LR = 0.01
DOWNLOAD_MINST = False

train_data = torchvision.datasets.MNIST(
    root  = './mnist',
    train = True,
    transform = torchvision.transforms.ToTensor(),  #np->Tensor ,(0, 1)
    download  = DOWNLOAD_MINST
)
# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(
    root  = './mnist/', 
    train = False,
)

print(train_data.data.size())         #torch.Size([60000, 28, 28])
print(train_data.targets.size())       #torch.Size([60000])
'''
plot first example

plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
'''

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)


test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                   # (batch, 32 ,7 ,7)                x.size() = torch.Size([50, 32, 7, 7])
        x = x.view(x.size(0), -1)           # 擴展成(batch_size, 32 * 7 * 7) , x.size() = torch.Size([50, 1568])
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
cnn.cuda()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
losses_history = []                                     # Plot LOSS

#he running mean is equivalent to convolving x with a vector that is N long, with all members equal to 1/N.
# remove the first N-1 points
def runningMeanFast(x, N):                              
    return np.convolve(x, np.ones((N,))/N ) [(N-1):]

plt.ion()
plt.show()
# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        batch_x = Variable(x).cuda()
        batch_y = Variable(y).cuda()

        output = cnn(batch_x)[0]               # cnn output
        loss = loss_func(output, batch_y).cuda()   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        losses_history.append(loss.cpu().data.numpy())
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            accuracy = float((pred_y == test_y.cpu().data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Step', step, '| Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.4f' % accuracy)
            plt.cla()
            plt.plot(runningMeanFast(losses_history,11),label='11 Points RA Loss', c='purple')
            plt.legend(loc='best')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.xlim((0, (60000*EPOCH/BATCH_SIZE)+200))
            plt.ylim((0, 1.2))
            plt.pause(0.1)
            
plt.ioff()
# print 50 predictions from test data
test_output, _  = cnn(test_x[:50])
pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()         #50 predictions
print(pred_y, 'prediction number')
print(test_y[:50].cpu().numpy(), 'real number')                            #50 solution