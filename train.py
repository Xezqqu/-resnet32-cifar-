from torchvision import transforms, datasets
import torch
import os
import json
from torch.utils.data import DataLoader
from model import resnet34
from tqdm import tqdm
import sys
import torchvision

def main():
    # 指定设备，device是str类型（字符串）
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device)) #将设备信息输出

    # 数据变换
    # 对CIFAR-10数据集进行预处理
    data_transfrom={
        "train":transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # CIFAR-10的图像大小为32x32
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # CIFAR-10的均值和标准差
        ]),
        "val":transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    }

    
    # 训练集
    train_data = torchvision.datasets.CIFAR10(
        root='./',  # 数据集将被下载到当前目录
        train=True,
        transform=data_transfrom["train"],
        download=True
    )
    train_num = len(train_data)

    # CIFAR-10的类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 保存类别信息
    class_dict = {i: classes[i] for i in range(10)}
    with open("class_index.json", "w") as json_file:
        json.dump(class_dict, json_file, indent=4)

    batch_size=128  # 增大批次大小，加快训练
    # 选择os.cpu_count()、batch_size、8中最小值作为num_workers
    nw=min([os.cpu_count(),batch_size if batch_size>1 else 0,8])
    print("Using {} dataloader workers every process".format(nw)) #打印使用几个进程

    # 设置训练集dataloader
    train_dataloader=DataLoader(train_data,batch_size=batch_size,
                                shuffle=True,num_workers=nw)
    train_num = len(train_data)
    # 测试集
    val_data = torchvision.datasets.CIFAR10(
        root='./',
        train=False,
        transform=data_transfrom["val"],
        download=True
    )
    val_num = len(val_data)  # 验证集长度

    # 设置验证集dataloader
    val_dataloader=DataLoader(val_data,batch_size=batch_size,
                              shuffle=False,num_workers=nw)

    print("Using {} for train,using {} for val".format(train_num,val_num))

    net=resnet34()

    # weigth_path=r"D:\SYS\ResNet-MobileNet-pytorch-cifar\resnet34-333f7ec4.pth"
    # assert os.path.exists(weigth_path),"weight file {} is not exists".format(weigth_path)
    # net.load_state_dict(torch.load(weigth_path,map_location=device,weights_only=False))

    # change fc layer structure，改变全连接层结构
    # CIFAR-10数据集有10个类别
    inchannels=net.fc.in_features
    net.fc=torch.nn.Linear(inchannels,10)
    net.to(device) #将模型放入设备（cpu或者GPU）中
    print(net.fc)
    
    # 构造优化器，使用SGD优化器
    params=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 损失函数，使用交叉熵损失
    loss_function=torch.nn.CrossEntropyLoss()

    epochs=200  # 增加训练轮次
    best_acc=0.0
    save_path="resnet34_cifar10.pth"
    train_step=len(train_dataloader)
    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss=0.0
        # file = sys.stdout的意思是，print函数会将内容打印输出到标准输出流(即 sys.stdout)
        # train_bar是tqdm用于显示进度
        train_bar=tqdm(train_dataloader,file=sys.stdout)
        data=train_data[0]
        for step,data in enumerate(train_bar):
            images,labels=data
            # images是一个batch的图片，[batcn_size,224,224]
            # labels是每个图片的标签，[batch_szie,]，如[1,0,4],数字代表类别
            optimizer.zero_grad() #先进行梯度清零
            pre=net(images.to(device)) #对类别进行预测,前向传播
            # print(pre)
            loss=loss_function(pre,labels.to(device))#计算损失
            loss.backward()#反向传播
            optimizer.step()#更新权重
            # loss 统计
            running_loss+= loss.item()

            train_bar.desc="train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

        #验证模式
        net.eval()
        acc=0.0 #预测正确个数
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_d in val_bar:
                val_image,val_label=val_d
                output=net(val_image.to(device))
                # torch.max比较后，第0个是每个最大值，第1个是最大值的下标，所以取第1个
                predict_y=torch.max(output,dim=1)[1]
                acc+=torch.eq(predict_y,val_label.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_step, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            
        # 更新学习率
        scheduler.step()
        
        # 打印当前学习率
        print(f'Current learning rate: {scheduler.get_last_lr()[0]:.6f}')
    
    print("Finished training! Best accuracy: {:.4f}".format(best_acc))



if __name__=="__main__":
    main()