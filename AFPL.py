from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy
#引入了alpha然后在每一个通信回合进行更新。
# 定义设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 定义训练函数
def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in tqdm(trainloader, desc="Training"):
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# 定义测试函数
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy






def load_data(partition_id, num_partitions=3):
    # 加载 CIFAR-10 数据集
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)

    # 计算每个分区的大小
    partition_size = len(dataset) // num_partitions
    print(partition_size)

    # 确定当前分区的索引范围
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else len(dataset)

    # 创建当前分区的子集
    partition = Subset(dataset, range(start_idx, end_idx))

    # 数据增强和归一化
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 应用转换
    partition.dataset.transform = pytorch_transforms

    # 使用 train_test_split 来分割训练集和测试集
    indices = list(range(len(partition)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # 根据索引创建训练和测试子集
    train_subset = Subset(partition, train_indices)
    test_subset = Subset(partition, test_indices)

    # 创建数据加载器
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32)

    return trainloader, testloader


# 示例：加载第 0 个分区的数据


# 联邦学习的核心步骤：模型参数聚合
def federated_average(models):
    #使用fedavg进行聚合
    averaged_params = OrderedDict()
    for key in models[0].keys():
        averaged_params[key] = sum(model[key] for model in models) / len(models)
    return averaged_params


# 联邦学习的主流程
def federated_learning(num_clients, num_rounds, num_epochs, a, learning_rate):
    global_model = Net().to(DEVICE)
    global_model.train()

    ten_model=Net().to(DEVICE)
    mylist=[a]*num_clients

    # 创建客户端的数据加载器
    client_loaders = [load_data(partition_id=i) for i in range(num_clients)]

    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1} / {num_rounds}")
        client_models = []
        Pa_List = []
        # 每个客户端进行本地训练
        for client_id, (trainloader, _) in enumerate(client_loaders):
            client_model = deepcopy(global_model).to(DEVICE)  #模拟分发
            train(client_model, trainloader, num_epochs)
            client_models.append(deepcopy(client_model.state_dict()))

        # 聚合客户端模型参数
        global_model_params = federated_average(client_models)
        global_model.load_state_dict(global_model_params)
        #分配和更新grada
        i=0
        for client_id, (trainloader, _) in enumerate(client_loaders):
            # 获取客户端和全局模型的状态字典
            local_params = client_models[client_id]
            global_params = global_model.state_dict()

            # 计算混合模型权重
            mixed_params = {
                key: mylist[i] * local_params[key] + (1 - mylist[i]) * global_params[key]
                for key in local_params.keys()
            }
            Pa_List.append(mixed_params)

            # 更新全局模型为混合模型
            ten_model.load_state_dict(mixed_params)

            # 测试混合模型并获取损失
            print("HYmODEL TEST")
            loss, accuracy = test(global_model, trainloader)
            print(f"Client {i}: Mixed Model Loss: {loss:.4f}, Mixed Model Accuracy: {accuracy:.4f}")

            # 计算相对于a的梯度并更新a
            grad_a = sum((local_params[key] - global_params[key]).sum().item() for key in local_params.keys())
            mylist[i] -= learning_rate * grad_a

            mylist[i] = max(0, min(1, a))
            i=i+1
        # 更新混合模型并测试
        i=0
        for client_id, (_, testloader) in enumerate(client_loaders):

            # 获取客户端和全局模型的状态字典
            local_params = client_models[client_id]
            global_params = global_model.state_dict()

            # 计算混合模型权重

            # 更新全局模型为混合模型
            ten_model.load_state_dict(Pa_List[i])

            # 测试混合模型并获取损失
            loss, accuracy = test(global_model, testloader)
            print(f"Client {client_id}: Mixed Model Loss: {loss:.4f}, Mixed Model Accuracy: {accuracy:.4f}")

            i=i+1


# 开始联邦学习
federated_learning(num_clients=3, num_rounds=5, num_epochs=10,a=0.5,learning_rate=0.1)
