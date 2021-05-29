from matplotlib import colors
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from   torch.autograd import Variable
import time

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态        , shape=(100, 2)
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)
print('x0:',x0)
print('x1:',x1)
"""========================================================================
torch.normal(mean, std, *, generator=None, out=None) → Tensor

std → standard deviation 定義： 
用於表示一組數值資料中的各數值相對於該組數值資料之平均數的分散程度。 
計算各數值與平均數的差，取其平方後加總，再除以數值個數，得「變異數」。 
變異數開根號後得「標準差」。

========================================================================"""
# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1),0 ).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1) ).type(torch.LongTensor)    # LongTensor = 64-bit integer

#plt.scatter( x軸數據 , y軸數據 , color , s->面積 , lw->linewidths , camp顏色範圍 )
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0,cmap='RdYlGn')
plt.show()

x , y = Variable(x) , Variable(y)

class Net(torch.nn.Module):                                                 #名為Net的class類別
    def __init__( self , n_feature , n_hidden , n_output ) :               #定義基礎宣告設定
        super( Net , self ).__init__()                                      #對繼承自父類的屬性進行初始化
        self.hidden  = torch.nn.Linear( n_feature  , n_hidden )     
        self.predict = torch.nn.Linear( n_hidden   , n_output )             #定義完這兩層不代表已經完成神經網路，要再下方的forward執行動作

    def forward( self , x ):                      #定義深度學習前傳的動作
        x = F.relu(self.hidden(x))
        predict_output = self.predict(x)
        return predict_output
    
net = Net(2 , 10 , 2)
plt.ion()   # 畫圖
plt.show()

optimizer = torch.optim.SGD( net.parameters() , lr = 0.01)
#對於分類、標籤誤差就利用torch.nn.CrossEntropyLoss() sEntropyLoss()
loss_func = torch.nn.CrossEntropyLoss()  

for t in range(500):
    out = net(x) #將輸出結果[a,b,c] 丟入softmax中 換算得到機率
    p   = F.softmax(out)
    loss = loss_func(out,y)
    optimizer.zero_grad()       #清空前一次的gradient
    loss.backward()             #根據loss進行back propagation，計算gradient
    optimizer.step()            #做梯度下降
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]    #[1]是最大值位置 [0]是最大值數值
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/2  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f '% accuracy+'%', fontdict={'size': 15, 'color':  'red'})
        plt.text(1.5, -3, 'Loss=%.4f '    % loss        , fontdict={'size': 15, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
