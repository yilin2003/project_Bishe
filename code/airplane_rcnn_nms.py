# 导入所需的库
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2  # OpenCV库，用于处理图像
import pandas as pd  # Pandas库，用于读取和处理表格数据（如CSV文件）
import matplotlib.pyplot as plt  # Matplotlib库，用于显示图像
import numpy as np  # Numpy库，处理数组数据

# 定义两个文件夹的路径
path = "Images"  # 存放图片文件的文件夹路径
annot = "Airplanes_Annotations"  # 存放标注信息（CSV文件）的文件夹路径

##整合前面两个步骤
# 初始化用于存储训练数据的列表
train_images = []  # 存储训练图片
train_labels = []  # 存储对应的标签 (1 表示飞机，0 表示非飞机)

# 定义计算 IOU（交并比）的函数
def get_iou(bb1, bb2):
    # 检查输入的边界框是否有效（左上角小于右下角）
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # 计算候选框和目标框的交集区域
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # 如果没有交集，返回 IOU = 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个边界框的面积
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # 根据交并比公式计算 IOU
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    # 检查 IOU 是否在 [0, 1] 范围内
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
# 创建 Selective Search 的分割器
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# 遍历标注文件夹中的所有标注文件
for e, i in enumerate(os.listdir(annot)):
    # 限制处理的文件数量为 100 个，加快调试和处理速度
    if e == 100:
        break

    try:
        # 只处理文件名前缀为 "airplane" 的标注文件
        if i.startswith("airplane"):
            # 根据标注文件名构建对应的图片文件名
            filename = i.split(".")[0] + ".jpg"
            print(e, filename)
            # 读取图片
            image = cv2.imread(os.path.join(path, filename))
            imout = image.copy()  # 复制图片以绘制候选框
            imout1 = image.copy()
            # 读取标注文件，包含目标框的位置信息
            df = pd.read_csv(os.path.join(annot, i))
            # 初始化存储目标框位置的列表
            gtvalues = []
            # 遍历标注文件中的每一行，将目标框保存到列表中
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                #print("目标框:",x1,x2,y1,y2)
            #     cv2.rectangle(imout1, (x1, y1), (x2 , y2), (255, 0, 0), 2)
            #     plt.figure()
            #     plt.imshow(imout1)
            # plt.show()


            # 使用 Selective Search 提取候选框
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()  # 快速模式
            ssresults = ss.process()  # 获取候选框
            print('number of ssresults:', ssresults.shape)


            # 初始化计数器和标志变量
            counter = 0  # 记录高 IOU 候选框的数量
            falsecounter = 0  # 记录低 IOU 候选框的数量
            flag = 0  # 控制是否停止处理候选框
            fflag = 0  # 标记是否达到高 IOU 候选框的上限
            bflag = 0  # 标记是否达到低 IOU 候选框的上限


            # 遍历候选框
            for e, result in enumerate(ssresults):
               # print('e,result:', e, result)
                yt = 0  # 为了展示原图中的候选框
                # 只处理前 2000 个候选框
                if e < 2000 and flag == 0:
                    # 遍历目标框
                    for gtval in gtvalues:
                        # 获取候选框的坐标和大小
                        x, y, w, h = result
                        #print("result:",result)
                        #print("x,y,w,h:",x, y, w, h)
                        # if(yt==0):
                        #     cv2.rectangle(imout1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        #     plt.figure()
                        #     plt.imshow(imout1)
                        #     yt=1

                        # 计算候选框和目标框的 IOU
                        iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})

                        # 如果 IOU > 0.7，将候选框视为正样本
                        if counter < 100:  # 限制最多保存 30 个正样本
                            if iou > 0.70:
                                timage = imout[y:y + h, x:x + w]  # 裁剪候选框区域
                               # print("正样本裁剪前")
                               #  plt.figure()
                               #  plt.imshow(timage)

                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)  # 调整尺寸
                                #使用 OpenCV 的 cv2.resize 方法，将裁剪后的图像调整为固定大小 224x224 像素。
                                #为了适配网络结构、降低计算成本，并统一输入尺度，确保训练的稳定性和效率。根据实际需求，输入大小也可以调整为其他尺寸（如 299x299 或 512x512），但需要与所选模型相匹配。
                                #插值方法选用 cv2.INTER_AREA，适合用于缩小图像，能保证更好的质量。
                                print("正样本:",iou)

                                # plt.figure()
                                # plt.imshow(resized)
                                # plt.show()
                                train_images.append(resized)  # 加入训练图片
                                train_labels.append(1)  # 标签为 1（表示飞机）
                                counter += 1
                        else:
                            fflag = 1  # 达到正样本上限

                        # 如果 IOU < 0.3，将候选框视为负样本
                        if falsecounter < 300:  # 限制最多保存 30 个负样本
                            if iou < 0.3:
                                timage = imout[y:y + h, x:x + w]  # 裁剪候选框区域
                                #print("负样本")
                                # plt.figure()
                                # plt.imshow(timage)

                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)  # 调整尺寸

                                print("负样本",iou)
                                # plt.figure()
                                # plt.imshow(resized)
                                # plt.show()
                                train_images.append(resized)  # 加入训练图片
                                train_labels.append(0)  # 标签为 0（表示非飞机）
                                falsecounter += 1
                        else:
                            bflag = 1  # 达到负样本上限

                    # 如果正负样本都达到上限，停止处理当前图片
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1

    except Exception as e:
        # 捕获异常并打印错误信息，跳过当前图片
        print(e)
        print("error in " + filename)
        continue

# 将训练数据转换为 NumPy 数组
X_new = np.array(train_images)  # 输入图片
y_new = np.array(train_labels)  # 对应标签
#将候选框送入CNN，提取出特征向量送入SVM识别，这里为了简单直接使用CNN识别,原文使用了AlexNet，这里使用了vgg16，vgg16的输入图片大小为224x224
import torch
X_new = torch.from_numpy(X_new)  # 将 numpy 数组转换为 PyTorch 张量
y_new = torch.from_numpy(y_new)  # 将标签转换为 PyTorch 张量

print("X_new.shape, y_new.shape=",X_new.shape, y_new.shape)  # 打印数据形状，检查是否正确
#batch_size传入网络中进行计算的样本数量，channels图像的通道数，中间特征的通道数
# 只检测图片中是否有飞机
#因为torch接受(b,c,w,h),所以更改维度
X_new = X_new.transpose(3, 1)  # 调整维度为 (batch_size, channels, height, width)

import torch
from torch import nn
# from torchvision.models import vgg16
#
# # 加载预训练的 VGG16 模型
# vgg = vgg16(pretrained=True)
from torchvision.models import vgg16, VGG16_Weights
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 使用权重参数替换 `pretrained=True`
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# 冻结 VGG16 模型的所有参数，使其在训练时保持不变
for p in vgg.parameters():
    p.requires_grad = False

for param in vgg.features[10:].parameters():  # 解冻最后一部分卷积层
    param.requires_grad = True

# vgg16的输入为224x224，定义一个新的神经网络类，继承自 nn.Module
class Mynet(nn.Module):#nn.Module: PyTorch 构建神经网络的基础类，提供了管理层次结构、参数跟踪、前向传播和模型保存的便利。
    def __init__(self):
        super(Mynet, self).__init__()#调用父类 nn.Module 的初始化方法
        self.vgg = vgg  # 使用预训练的 VGG16 模型作为特征提取器
        # 定义线性分类层（全连接层）
        self.fc1 = nn.Linear(1000, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

        # self.fc1 = nn.Linear(1000, 512)  # 第一层输入为 VGG16 的输出 (1000)，输出为 512
        # self.fc2 = nn.Linear(512, 256)  # 第二层，输入为 512，输出为 256
        # self.fc3 = nn.Linear(256, 256)  # 第三层，输入为 256，输出为 256
        # self.fc4 = nn.Linear(256, 10)  # 第四层，输入为 256，输出为 10
        # self.fc5 = nn.Linear(10, 2)  # 最后一层，输入为 10，输出为 2（分类任务：二分类）
        self.relu = nn.ReLU()  # 激活函数使用 ReLU
        self.softmax = nn.Softmax(dim=1)  # 最后一层使用 Softmax 生成概率分布

    # 定义前向传播函数
    def forward(self, x):
        x = vgg(x)                  # 使用 VGG16 提取特征
        x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU 激活
        x = self.relu(self.fc2(x))  # 第二层全连接 + ReLU 激活
        x = self.relu(self.fc3(x))  # 第三层全连接 + ReLU 激活
        x = self.relu(self.fc4(x))  # 第四层全连接 + ReLU 激活
        x = self.softmax(self.fc5(x))  # 最后一层全连接 + Softmax 激活
        #全连接层 fc5 将 10 维特征降维到 2 维，对应二分类任务（如：有飞机/无飞机）。
        #使用 Softmax 函数，将 2 维向量转化为概率分布，表示属于每一类的概率
        return x
# 使用训练数据和标签创建 Tensor 数据集
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_new, y_new)

# 定义数据加载器，按批次加载数据
dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,  # 每批次 64 个样本
    shuffle=True,   # 打乱数据顺序
    num_workers=0   # 不使用多线程加载数据
)
# 实例化自定义网络
#net = Mynet()
# 将模型移到 GPU
net = Mynet().to(device)


# 定义优化器，使用 Adam 优化算法，学习率为 0.0005
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
# 定义损失函数，使用交叉熵损失（适用于分类任务）
criterion = nn.CrossEntropyLoss()#对预测的概率值取对数，然后计算与实际类别的负对数似然
# 开始训练模型
for i, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)
    pred = net(x.to(torch.float32))   # 保输入数据 x 转换为浮点数类型 torch.float32，并通过网络计算预测值,要求输入为 32 位浮点数（尤其是预训练的模型如 VGG16）。
    loss1 = criterion(pred, y.long())  # 将 y 转换为 long 类型
      # 计算损失值（预测值与真实标签的差异）
    #print("i,loss=",i, loss1.item())           # 打印当前批次的损失值
    optimizer.zero_grad()            # 清空优化器的梯度
    loss1.backward()                 # 反向传播计算梯度
    optimizer.step()                 # 更新模型参数
   # print("x.shape, y.shape, pred.shape:",x.shape, y.shape, pred.shape)  # 可选：打印输入、标签和预测值的形状
#预测
z=0
for e1,i in enumerate(os.listdir(path)):
#.  z<10为了早点结束
    if(z==10):
        break
    if i.startswith("4"):
        z += 1
        img = cv2.imread(os.path.join(path, i))  # 读取目标图片
        ss.setBaseImage(img)  # 设置图片为选择性搜索的基底图像
        #ss.switchToSelectiveSearchFast()  # 启用快速模式，生成候选框
        ss.switchToSelectiveSearchQuality()  # 切换到更精确的模式，可能会减少候选框数量

        ssresults = ss.process()  # 获取所有候选框
        imout = img.copy()  # 创建图片副本，用于绘制结果


        import torchvision.ops as ops

        import torchvision.ops as ops
        import torch


        # 对候选框应用 NMS
        def apply_nms(boxes, scores, threshold=0.9):
            # 确保数据类型为 float
            boxes = boxes.to(torch.float)
            scores = scores.to(torch.float)
            keep = ops.nms(boxes, scores, threshold)
            return keep


        def compute_iou(box, other_boxes):

            # 计算交集
            x_min = torch.max(box[0], other_boxes[:, 0])
            y_min = torch.max(box[1], other_boxes[:, 1])
            x_max = torch.min(box[2], other_boxes[:, 2])
            y_max = torch.min(box[3], other_boxes[:, 3])

            inter_area = torch.clamp(x_max - x_min, min=0) * torch.clamp(y_max - y_min, min=0)

            # 计算并集
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            other_boxes_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            union_area = box_area + other_boxes_area - inter_area

            # 计算 IoU
            ious = inter_area / torch.clamp(union_area, min=1e-6)  # 避免除零
            return ious


        def process_and_nms(candidate_boxes, scores, iou_threshold=0.5):
            """
            对候选框集合执行非极大值抑制 (NMS)。
            """
            sorted_indices = torch.argsort(scores, descending=True)
            keep_indices = []  # 存储保留下来的框索引

            while len(sorted_indices) > 0:
                current = sorted_indices[0]  # 最高分的框
                keep_indices.append(current.item())

                if len(sorted_indices) == 1:
                    break

                # 计算 IoU
                ious = compute_iou(candidate_boxes[current], candidate_boxes[sorted_indices[1:]])

                # 保留 IoU 小于阈值的框
                remaining = torch.where(ious <= iou_threshold)[0]
                sorted_indices = sorted_indices[remaining + 1]  # 跳过已处理的框

            return keep_indices


        all_boxes = []
        all_scores = []

        # 控制处理的候选框数量为前 500 个框
        for e, result in enumerate(ssresults[:100]):  # 只处理前 500 个候选框

            x, y, w, h = result  #
            timage = imout[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            img = torch.from_numpy(img).permute(0, 3, 1, 2)
            # 确保张量类型为 float32，并移动到 GPU
            img = img.to(device).to(torch.float32)
            out = net(img)
            print("out=", out)

            # # 在绘制框之前，判断置信度并绘制不同颜色的框
            # if out[0][0] > out[0][1] and out[0][0] > 0.9:
            #     score = out[0][0].item()
            #     boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float)
            #     scores = torch.tensor([score], dtype=torch.float)
            #     print("scores:",scores)
            #     print("boxes :", boxes )
            #     iou_thresh = 0.1
            #     keep = apply_nms(boxes, scores, threshold=0.9)
            #     #keep=non_max_suppression(boxes, scores, iou_threshold=iou_thresh)
            #     cv2.rectangle(imout, (x, y), (x + w, y + h), (0,255,0), 1, cv2.LINE_AA)
            #     plt.figure()
            #     plt.imshow(imout)
            #     plt.show()
            # 如果是目标类（例如背景与物体二分类），根据置信度进行筛选
            # 确保所有分数都在 all_scores 中累积
            #all_scores = torch.tensor(all_scores, dtype=torch.float32)  # 转为 PyTorch 张量

            # 计算分数阈值
           # score_threshold = all_scores.mean().item() if len(all_scores) > 0 else 0.7

           # print("score_threshold =", score_threshold)

            if out[0][0] > 0.9:
                score = out[0][0].item()  # 转为 float
                box = [x, y, x + w, y + h]  # 转为 [x_min, y_min, x_max, y_max] 格式

                # 将当前框及分数加入列表
                all_boxes.append(box)
                all_scores.append(score)

        # 转换为张量
        all_boxes = torch.tensor(all_boxes, dtype=torch.float32)
        all_scores = torch.tensor(all_scores, dtype=torch.float32)

        iou_thresh = 0.3  # 提高 IoU 阈值

        # 执行 NMS
        if len(all_boxes) > 0:
            all_boxes = torch.tensor(all_boxes, dtype=torch.float32)
            all_scores = torch.tensor(all_scores, dtype=torch.float32)

            keep_indices = process_and_nms(all_boxes, all_scores, iou_threshold=iou_thresh)

            # 绘制筛选后的框
            for idx in keep_indices:
                box = all_boxes[idx].int()
                score = all_scores[idx].item()
                x_min, y_min, x_max, y_max = box.tolist()
                areas = [(x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max in all_boxes]
                min_area = np.percentile(areas, 10)  # 保留面积在前 10% 的最小值
                max_area = np.percentile(areas, 90)  # 保留面积在前 90% 的最大值

                # 过滤面积
                box_area = (x_max - x_min) * (y_max - y_min)
                if box_area > min_area and box_area < max_area:
                    # 根据置信度调整颜色
                    color = (0, int(255 * score), 0)  # 绿色随置信度变化
                    cv2.rectangle(imout, (x_min, y_min), (x_max, y_max), color, 2, cv2.LINE_AA)

        # 显示结果
        plt.figure(figsize=(10, 10))
        plt.imshow(imout)
        plt.axis('off')
        plt.show()
