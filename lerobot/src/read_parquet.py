from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 自动加载本地或云端数据集
dataset = LeRobotDataset(r"/home/yueteng/.cache/huggingface/lerobot/yueteng/pusht1202_3")

# 1. 查看数据集信息
print(f"总帧数: {dataset.num_frames}")
print(f"包含的特征: {dataset.features}")

# 2. 获取某一帧的数据 (会自动解码视频帧)
frame_index = 10
item = dataset[frame_index]

# item 是一个字典，包含当前时刻的所有信息
# 例如：
# item['observation.state'] -> 机械臂当前状态
# item['action'] -> 对应的动作
# item['observation.images.phone'] -> 如果有图像，这里是解码后的 Tensor
print(item.keys())