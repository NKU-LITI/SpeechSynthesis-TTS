import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    """
    根据给定的长度生成一个掩码张量。
    参数：lengths (Tensor): 包含每个序列长度的张量。
    返回值：mask (Tensor): 一个布尔掩码张量，其中 True 表示有效的时间步长，False 表示填充的时间步长。
    """
    # 获取长度张量中的最大值
    max_len = torch.max(lengths).item()
    
    # 创建一个从0到max_len的长整型张量
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    
    # 生成掩码，长度小于相应位置的元素为True，反之为False
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

def load_wav_to_torch(full_path):
    """
    从文件中加载音频数据并转换为PyTorch张量。
    参数：full_path (str): 音频文件的完整路径。
    返回值：Tuple (Tensor, int): 返回一个包含音频数据的浮点型张量和采样率的元组。
    """
    # 读取音频文件，返回采样率和数据
    sampling_rate, data = read(full_path)
    
    # 将数据转换为浮点型张量
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
    """
    从文件中加载文件路径和文本对。

    参数：
    filename (str): 包含文件路径和文本对的文件名。
    split (str): 分隔符，默认为 "|"

    返回值：
    List: 包含文件路径和文本对的列表。
    """
    # 打开文件并读取内容
    with open(filename, encoding='utf-8') as f:
        # 将每一行按分隔符拆分并去掉首尾空白字符，生成文件路径和文本的列表
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def to_gpu(x):
    """
    将张量移动到GPU上（如果可用）。
    params：x (Tensor): 需要移动到GPU的张量。
    return：Variable: 位于GPU上的张量（如果可用），否则返回原张量。
    """
    # 保持张量在内存中的连续性
    x = x.contiguous()

    # 如果GPU可用，将张量移动到GPU上
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    
    # 将张量包装为自动求导变量
    return torch.autograd.Variable(x)
