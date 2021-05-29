from matplotlib import colors
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from   torch.autograd import Variable
import time

x = torch.unsqueeze(torch.linspace(-1, 1, 500), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())               # x.size() ==> shape=(100, 1)
x , y = Variable(x) , Variable(y)                       #神經網路只接受variable的輸入

#繪圖
#plt.scatter( x, y, color='gray')
#plt.show()
plt.ion()   # 画图
plt.show()

class Net(torch.nn.Module):                                                 #名為Net的class類別
    def __init__( self , n_feature , n_hidden , n_output ) :               #定義基礎宣告設定
        super( Net , self ).__init__()                                      #對繼承自父類的屬性進行初始化
        self.hidden  = torch.nn.Linear( n_feature  , n_hidden )     
        self.predict = torch.nn.Linear( n_hidden   , n_output )             #定義完這兩層不代表已經完成神經網路，要再下方的forward執行動作

    def forward( self , x ):                      #定義深度學習前傳的動作
        x = F.relu(self.hidden(x))
        predict_output = self.predict(x)
        return predict_output
    
net = Net(1 , 10 , 1)
#print(net)  印出有多少連接層

optimizer = torch.optim.SGD( net.parameters() , lr = 0.1)
loss_func = torch.nn.MSELoss(reduction='mean')

for t in range(500):
    prediction = net(x)

    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()