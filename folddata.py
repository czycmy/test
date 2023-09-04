import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
import cv2
import os
from model import unet
from torch import optim
import torch.nn as nn

device = torch.device('cuda')
print(device)




# 保存网络参数
save_path = './UNet.pth'  # 网络参数的保存路径
best_acc = 0.0  # 保存最好的准确率
test_jaccards = []

Epoch = 60

# Set up batch size
batch_size = 4
# Root directory to dataset
root_dir = './Dataset_BUSI'
# Instantiate root directory of dataset to path
path = Path(root_dir)
# Creating list of both image and mask paths
image_paths = list(path.glob('*.png'))
# Creating list of image paths
images = [str(image_path) for image_path in image_paths if '_mask' not in str(image_path)]
# Creating list of mask paths
masks = [str(image_path) for image_path in image_paths if '_mask' in str(image_path)]

transforms_image = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transforms_masks = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((224, 224))])


class CustomDataset(Dataset):
    def __init__(self, images: list, masks: list, transforms1, transforms2):
        # Store image and mask paths as well as transforms
        self.images = images
        self.masks = masks
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __getitem__(self, index):
        # Capture image and mask path from the current index
        image_path = self.images[index]
        mask_path = self.masks[index]
        # Load image
        image = cv2.imread(image_path)
        # Swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read the respective mask
        mask = cv2.imread(mask_path, 0)
        # Check to see if we are applying any transforms
        if self.transforms1 is not None:
            # Performing transforms to image and mask
            images = self.transforms1(image)
            masks = self.transforms2(mask)
        # Return tuple of images and their respective masks
        return images, masks

    def __len__(self):
        # Return the number of total images contained in the dataset
        return len(self.images)


def dice_coef(output, target):  # batch_size=1
    smooth = 1e-5
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = torch.sigmoid(output)
    output[output > 0.5] = 1  # 将概率输出变为于标签相匹配的矩阵
    output[output <= 0.5] = 0
    # target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()
    # \符号有换行的作用
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def get_sensitivity(output, gt):
    # 求敏感度 se=TP/(TP+FN)
    SE = 0.
    output = output > 0.5
    gt = gt > 0.5
    TP = ((output == 1).byte() + (gt == 1).byte()) == 2
    FN = ((output == 0).byte() + (gt == 1).byte()) == 2

    if len(output) > 1:
        for i in range(len(output)):
            SE += float(torch.sum(TP[i])) / (float(torch.sum(TP[i] + FN[i])) + 1e-6)
    else:
        SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold
    SP = 0.
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    if len(SR) > 1:
        for i in range(len(SR)):
            SP += float(torch.sum(TN[i])) / (float(torch.sum(TN[i] + FP[i])) + 1e-6)
    else:
        SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def precision(output, target):
    # 将输出和目标转换为二进制张量
    outputs = torch.round(output)
    targets = torch.round(target)

    # 计算真阳性（True Positives）
    true_positives = torch.sum(torch.logical_and(outputs == 1, targets == 1)).item()

    # 计算假阳性（False Positives）
    false_positives = torch.sum(torch.logical_and(outputs == 1, targets == 0)).item()

    # 计算精确度
    precision = true_positives / (true_positives + false_positives + 1e-7)

    return precision


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1) * 1 + (GT == 1) * 1) == 2
    FP = ((SR == 1) * 1 + (GT == 0) * 1) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def Iou(pred, true):
    intersection = pred * true  # 计算交集  pred ∩ true
    temp = pred + true  # pred + true
    union = temp - intersection  # 计算并集：A ∪ B = A + B - A ∩ B
    smooth = 1e-8  # 防止分母为 0
    iou_score = intersection.sum() / (union.sum() + smooth)
    return iou_score


# Creating K-fold cross-validation splits
k_folds = 4
kf = KFold(n_splits=k_folds, shuffle=True, random_state=12)
splits = kf.split(images)

list2 = 0.0
list3 = 0.0
list4 = 0.0
list5 = 0.0


# Loop through each fold and train the model
for fold, (train_indices, val_indices) in enumerate(splits):
    net = unet(1)  # 加载网络
    net.to(device)  # 将网络加载到device上
    optimizer = optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)  # 定义优化器
    criterion = nn.BCEWithLogitsLoss()  # 定义损失函数
    # Splitting images and their respective masks into train and validation sets for the current fold
    print('------------------------------------------------------------------------------------------')
    print(f"Fold {fold + 1}")
    train_data = [images[i] for i in train_indices]
    val_data = [images[i] for i in val_indices]
    train_data_masks = [masks[i] for i in train_indices]
    val_data_masks = [masks[i] for i in val_indices]

    # Creating train and validation Datasets for the current fold
    train_dataset = CustomDataset(images=train_data, masks=train_data_masks, transforms1=transforms_image,
                                  transforms2=transforms_masks)
    val_dataset = CustomDataset(images=val_data, masks=val_data_masks, transforms1=transforms_image,
                                transforms2=transforms_masks)

    # Creating train and validation DataLoaders for the current fold
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    num1 = len(train_dataset)
    num2 = len(val_dataset)

    print('new fold start:')
    print('-------------------------------------------------------------------------------------------')

    # Training and evaluation for the current fold
    for epoch in range(Epoch):

        net.train()  # 训练模式
        running_loss = 0.0

        for (images_batch, masks_batch) in train_loader:
            optimizer.zero_grad()  # 梯度清零
            pred = net(images_batch.to(device))  # 前向传播
            loss = criterion(pred, masks_batch.to(device))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降

            running_loss += loss.item()  # 计算损失和

        net.eval()  # 测试模式
        acc = 0.0  # 正确率
        total = 0
        test_running_loss = 0.0
        dice_coeff = 0.0
        precision = 0.0
        total_samples = 0
        total_dice = 0
        tsample = 0
        sensitivity_sum = 0.0
        specificity_sum = 0.0
        ppv_sum = 0.0
        total_jaccard = 0.0
        running_jaccard = 0
        with torch.no_grad():
            for (images_batch, masks_batch) in val_loader:
                bs = images_batch.size(0)
                outputs = net(images_batch.to(device))  # 前向传播

                test_loss = criterion(outputs, masks_batch.to(device))
                test_running_loss += test_loss.item()

                predicted_masks = torch.sigmoid(outputs) > 0.5  # 根据输出进行二值化处理
                predicted_masks = predicted_masks.float().to(device)
                masks_batch = masks_batch.to(device)

                # 计算Dice系数
                intersection = torch.sum(predicted_masks * masks_batch).to(device)
                union = torch.sum(predicted_masks) + torch.sum(masks_batch)
                dice_coeff += (2 * intersection) / (union + 1e-8)

                # 计算准确率（Precision）
                true_positives = torch.sum(predicted_masks * masks_batch)
                predicted_positives = torch.sum(predicted_masks)
                precision += true_positives / (predicted_positives + 1e-8)

                total_samples += images_batch.size(0)

                dice = dice_coef(outputs, masks_batch)
                total_dice += dice * bs
                tsample += bs

                sensitivity = get_sensitivity(outputs, masks_batch)
                sensitivity_sum += sensitivity

                specificity = get_specificity(outputs, masks_batch)
                specificity_sum += specificity

                ppv = get_precision(outputs, masks_batch)
                ppv_sum += ppv * bs



                outputs[outputs >= 0] = 1  # 将预测图片转为二值图片
                outputs[outputs < 0] = 0

                # 计算预测图片与真实图片像素点一致的精度：acc = 相同的 / 总个数
                acc += (outputs == masks_batch.to(device)).sum().item() / (masks_batch.size(2) * masks_batch.size(3))
                total += masks_batch.size(0)


        accurate = acc / total  # 计算整个test上面的正确率
        loss1 = test_running_loss / len(val_loader)
        avg_dice_coeff = total_dice / tsample
        avg_precision = precision / total_samples
        average_sensitivity = sensitivity_sum / num2
        average_specificity = specificity_sum / num2
        average_ppv = ppv_sum / tsample


        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f  test_loss: %.3f  avg_dice_coeff: %.3f ' %
              (epoch + 1, running_loss / num1, accurate * 100, loss1, avg_dice_coeff * 100))
        print("precision（准确率）:{:.3f}".format(average_ppv))
        print("specificity(特异性):{:.3f}".format(average_specificity))
        print("recall(召回率):{:.3f}".format(average_sensitivity))

        if epoch+1 == Epoch:
            list2 += avg_dice_coeff
            list3 += average_ppv
            list4 += average_specificity
            list5 += average_sensitivity


    if fold == 3:
        b2 = list2 / k_folds
        c3 = list3 / k_folds
        d4 = list4 / k_folds
        e5 = list5 / k_folds
        print("------------------------------------------------------------------------------------")
        print("dice_coeff:{:.3f}".format(b2))
        print("precision（准确率）:{:.3f}".format(c3))
        print("specificity(特异性):{:.3f}".format(d4))
        print("recall(召回率):{:.3f}".format(e5))









        # if accurate > best_acc:  # 保留最好的精度
        #     best_acc = accurate
        #     torch.save(net.state_dict(), save_path)  # 保存网络参数
    # Your training and evaluation code here

