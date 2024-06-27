from collections import OrderedDict
import torch.nn as nn
from utils import *

class MLP(nn.Module):
    def __init__(self, units: list):
        super(MLP, self).__init__()
        self.units = units # contain the input_dim
        self.hidden_numbers = len(self.units) - 1

        layers = []
        for i in range(self.hidden_numbers):
            layers.extend([nn.Linear(self.units[i], self.units[i + 1]), nn.BatchNorm1d(units[i + 1]), nn.Tanh()])
        self.backbone_net = nn.ModuleList(layers)
        self.backbone_net = nn.Sequential(*self.backbone_net)

    def forward(self, x):
        z = self.backbone_net(x)
        return z

class Fusion(nn.Module):
    def __init__(self, fusion_dim=1024, nbit=64) -> None:
        super(Fusion, self).__init__()
        self.hash = nn.Sequential(
            nn.Linear(fusion_dim, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh())

    def forward(self, x, y):
        hash_code = self.hash(x + y)
        return hash_code

# 定义Self-Attention模块
class SelfAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(memory_dim, hidden_dim)
        self.value = nn.Linear(memory_dim, hidden_dim)

    def forward(self, query, key, value):
        query = self.query(query)
        key = self.key(key.to(self.key.weight.dtype))
        value = self.value(value.to(self.value.weight.dtype))

        attention_scores = (query @ key.T) / math.sqrt(self.hidden_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_input = attention_weights @ value

        return weighted_input

# 定义文本编码器
class TextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TextEncoder, self).__init__()
        self.textEncoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

    def forward(self, text_input):
        encoded_text = self.textEncoder(text_input)
        return encoded_text

# 定义图像编码器
class ImageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ImageEncoder, self).__init__()
        self.imageEncoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

    def forward(self, image_input):
        encoded_image = self.imageEncoder(image_input)
        return encoded_image

# 定义文本解码器
class TextDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(TextDecoder, self).__init__()
        self.textDecoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

    def forward(self, text_input):
        decoded_text = self.textDecoder(text_input)
        return decoded_text

# 定义图像解码器
class ImageDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(ImageDecoder, self).__init__()
        self.imageDecoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

    def forward(self, image_input):
        decoded_image = self.imageDecoder(image_input)
        return decoded_image

# 定义文本缺失补全模型
class TextCompletionModel(nn.Module):
    def __init__(self, input_dim, map_dim, hidden_dim, memory_dim, output_dim, memory_size):
        super(TextCompletionModel, self).__init__()
        self.text_memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_dim = memory_dim
        self.image_encoder = ImageEncoder(input_dim, map_dim)
        self.attention = SelfAttention(map_dim, memory_dim, hidden_dim)
        self.text_decoder = TextDecoder(hidden_dim, output_dim)

    def forward(self, image_input):
        image_input_encode = self.image_encoder(image_input)
        attended_input = self.attention(image_input_encode, self.text_memory, self.text_memory)
        generated_text = self.text_decoder(attended_input)
        return generated_text

# 定义图像缺失补全模型
class ImageCompletionModel(nn.Module):
    def __init__(self, input_dim, map_dim, hidden_dim, memory_dim, output_dim, memory_size):
        super(ImageCompletionModel, self).__init__()
        self.image_memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_dim = memory_dim
        self.attention = SelfAttention(map_dim, memory_dim, hidden_dim)
        self.text_encoder = TextEncoder(input_dim, map_dim)
        self.image_decoder = ImageDecoder(hidden_dim, output_dim)

    def forward(self, text_input):
        text_input_encode = self.text_encoder(text_input)
        attended_input = self.attention(text_input_encode, self.image_memory, self.image_memory)
        generated_image = self.image_decoder(attended_input)
        return generated_image