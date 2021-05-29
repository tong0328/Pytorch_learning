import torch
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable

# torch.manual_seed(1)    # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.1)  #Stochastic  Gradient Descent (SGD)
    loss_func = torch.nn.MSELoss()

    for times in range(1000):
        prediction = net1(x)
        loss       = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1,'net1.pkl')                       #save entire net
    torch.save(net1.state_dict(),'net1_params.pkl')   #save all parameters in net1

    #plot result
    plt.figure(1, figsize=(10,3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter( x.data.numpy(), y.data.numpy() )
    plt.plot( x.data.numpy(), prediction.data.numpy() ,'r-', lw=5)
    

def restore_net():
    start = time.time()                             # compute how many time is the params model need.
    net2 = torch.load('net1.pkl')
    prediction = net2(x)
    #plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter( x.data.numpy(), y.data.numpy() )
    plt.plot( x.data.numpy(), prediction.data.numpy() ,'r-', lw=5)
    
    print("store net:",time.time() - start)         # compute how many time is the params model need.

    

def restore_net_params():
    start = time.time()                             # compute how many time is the params model need.
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict( torch.load('net1_params.pkl') )
    prediction = net3(x)
     #plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter( x.data.numpy(), y.data.numpy() )
    plt.plot( x.data.numpy(), prediction.data.numpy() ,'r-', lw=5)  
    print("store net's params:",time.time()-start)  # compute how many time is the params model need.
save()
restore_net()
restore_net_params()
plt.show()