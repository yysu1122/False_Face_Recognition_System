import time 
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split,ConcatDataset 
from torchvision import datasets 
import matplotlib.pyplot as plt 
plt.ion()


tarDirTrain = "test2/CNN_synth_testset/train" 
tarDirTest = "test2/CNN_synth_testset/test"

#预处理
normalize=transforms.Normalize(mean=[.5, .5,.5],std=[.5,.5,.5])
#normalize=transforms.Normalize([0.485，0.456，0.406]，[0.229，0.224，0.225]) 
transform=transforms.Compose([#transforms.RandomReSizedCrop(224)，
    transforms.RandomHorizontalFlip(),#水平翻转
    transforms.RandomRotation(10),	#随机旋转	
    transforms.ToTensor(),#将图片转换为Tensor,归一化至[0,1]
    normalize	#[-1,1]
])


training_data = datasets.ImageFolder(tarDirTrain, transform =transform) 
test_data = datasets.ImageFolder(tarDirTest,transform = transform)

batch_size= 16 #加载批次
num_workers = 4#设置四个进程加载

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers= num_workers) 
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=True, num_workers= num_workers)

class_names = training_data.classes	#给训练图像分类标签	
class_idx = training_data.class_to_idx#分别对应索引0，1


def imageshow(imgs,labels):
    images_so_far =0
    for j in range(len(imgs)):
        images_so_far += 1
        plt.subplot(1,len(imgs),images_so_far)
        plt.title( f"{class_names[labels[j]]}",fontsize=10,color = "b") 
        img = imgs[j].numpy().transpose(1,2,0)
        img = img / 2 +0.5	# unnormalize 归一化操作	
        plt.imshow(img) 
    plt.ioff() 
    plt.show()


# #测试显示一些图片
# #get some random training image 
# if __name__ == '__main__':
#     dataiter = iter(train_dataloader)
#     images, labels = dataiter.next()#这是一个迭代器。# show images
#     imageshow(images, labels)# print labels
#     print(''.join('%5s'% class_names[labels[j]] for j in range(8)))



#定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init_()
        self.conv0 = nn.Conv2d(3,30,3)#输入通道数3，输出通道数为30，卷积核为3x3大小256- 2 =254 
        self.batch1 = nn.BatchNorm2d(30)
        self.pool1= nn.MaxPool2d(2,2)	#最大池化层	254 / 2=127	
        self.conv1 = nn.Conv2d(30,90 ,4)	#127 -4 +1 =124	
        self.batch2 = nn.BatchNorm2d(90)
        self.conv2 = nn.Conv2d(90,30,5)	#(124 - 5)+1=120	120/2=60	
        self.batch3 = nn.BatchNorm2d(30)
        self.pool2= nn.MaxPool2d(2,2)	#60	

        self.conv3 = nn.Conv2d(30,16,3)#60-3 +1 =58 
        self.batch4 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,16,3)	#58 - 3 +1 =56	
        self.batch5 = nn.BatchNorm2d(16)
        self.pool3 =nn.MaxPool2d(2,2)	#56/2=28

        self.conv5 =nn.Conv2d(16,16,3)	#28 -3 + 1 =26	
        self.batch6 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16,16,3)	##24	
        self.batch7 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(2,2)	#12	

        self.conv7 = nn.Conv2d(16,16 ,3) #12-2=10 
        self.batch8 = nn.BatchNorm2d(16)
        self.pool5=nn.MaxPool2d(2,2) #10/2 =5
        self.fc1 = nn.Linear(16* 5 * 5,200)#全连接层，必须将[16，5，5]先view()成16*5*5才能使用全连接层 
        self.fc2 = nn.Linear(200,120) 
        self.fc3 = nn.Linear(120,2)
        #self.drop1 =nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.pool1(F.relu(self.batch1(self.conve(x) )))
        x = F.relu(self.batch2(self.conv1(x))) 
        x = F.relu(self.batch3(self.conv2(x))) 
        x = self.pool2(x)

        x = F.relu(self.batch4(self.conv3(x))) 
        x = F.relu(self.batch5(self.conv4(x))) 
        x = self.pool3(x)

        x = F.relu(self.batch6(self.conv5(x))) 
        x = F.relu(self.batch7(self.conv6(x))) 
        x = self.pool4(x)

        x = F.relu(self.batch8(self.conv7(x))) 
        x = self.pool5(x)

        x = x.view(-1,16 * 5 * 5) #x.size()[0]，-1

        #x = self.drop1(x)
        x = F.relu(self.fc1(x)) 
        x =F.relu(self.fc2(x)) 
        x = self.fc3(x)
        #x = F.softmax(x, dim = 1)	#按照行做归一化log softmax就是在softmax基础上再做一次1og运算	
        return x


# #获取用于训练的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device" )
net = Net().to(device)	#实例化网络模型


#定义损失函数和优化器
#损失函数为交叉熵损失函数
#采用的是优化器是随机梯度下降优化器(是包含动量部分的) 
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001, momentum=0.9)


#训练
def train(dataloader,model, loss_fn, optimizer):
    size = len(dataloader.dataset) 
    model.train()
    train_loss,correct = 0,0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error 
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()#损失值和准确率
        train_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        #if batch % 100 == 0: #每100
        #	loss,current =loss.item( )，batch * len(X)	
        #   print(f"loss: {loss:>7f}[{current:>5d}/{size:>5d}]")
        #	loss_plot.append(train_loss / 100)	
        #   train_loss =0.0
    train_loss /= size 
    correct /= size
    print(f"Train Error:\n Accuracy:{(100*correct):>0.1f}%,Avg loss: {train_loss:>8f} \n")
    return model, train_loss,correct


#测试训练集的函数
def test(dataloader,model,loss_fn):

    size = len(dataloader.dataset) 
    num_batches = len(dataloader ) 
    model.eval()
    test_loss,correct = 0,0 
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device),y.to(device) 
            pred = model(X)
            test_loss += loss_fn(pred, y).item( )
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches 
    correct /= size
    print(f"Test Error: \n Accuracy:{(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return model,correct


#预测测试集中的数据并可视化图像
def visualize_model(model,dataloader, num_images=8):
    was_training = model.training 
    model.eval()
    images_so_far =0 
    with torch.no_grad():
        for i,(inputs, labels) in enumerate(dataloader):	#先在训练集	
            inputs = inputs.to(device) 
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//4, 4,images_so_far) 
                ax.axis('off' )
                ax.set_title(f"pred:{class_names[preds[j]]}\n true:{class_names[labels[j]]}", \
                    fontsize=10, color='green'if preds[j]== labels[j] else 'red') 
                img = inputs.cpu().data[j] 
                # img = img.swapaxes(0,1)	#换成256*256*3	
                # img = img.swapaxes(1,2)
                # img = img * 0.5 +0.5 #还原
                img = img / 2 + 0.5	# unnormalize 归一化操作	
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg,(1,2,0)))# plt.imshow(img)
                if images_so_far == num_images:
                    model.train(mode=was_training) 
                    return
        model.train(mode=was_training)


loss_plot =[] 
train_acc = [] 
test_acc = [] 
epochs = 150
# best_model_wts = copy.deepcopy(net.state_dict()) best_acc = 0.0
for t in range(epochs):
    print(f"Epoch {t+1}\n---- ")
    since = time.time()
    net, train_loss, correct = train(train_dataloader, net, loss_func, optimizer) 
    loss_plot.append(train_loss) 
    train_acc.append(correct)

    #net,epoch_acc = test(train_dataloader, net, loss_func)
    # #测试训练集中的准确率#利用当前模型预测测试集的数据
    net,epoch_acc = test(test_dataloader, net, loss_func) 
    test_acc.append(epoch_acc)

    #更新最好的模型状态
    # if epoch acc > best_acc:
    #	best_acc = epoch_acc	
    #	best_model_wts = copy.deepcopy(net.state_dict())	
    #时间
    time_elapsed =time.time() - since
    print(f'Training complete in {time_elapsed // 60: .0f}m {time_elapsed % 60: .0f}s')
    # net.load state dict(best model wts)


    #如果在训练集的分类准确率到1，则停止训练 
    if correct == 1:
        break
#看一下损失函数的变化 
plt.plot(loss_plot)

#准确率的变化
plt.plot(train_acc) 
plt.plot(test_acc)


#保存
PATH ='./fakeFace_net.pth'
torch.save(net.state_dict(),PATH)


#加载训练数据集
# PATH = './fakeFace_net.pth'
# net.load_state_dict(torch.load(PATH))
#验证测试集准确率
test_loss,correct = test(test_dataloader, net,loss_func)
#使用测试集进行预测并可视化
visualize_model(net,test_dataloader)

plt.ioff() 
plt.show( ) 
print("Done!")


