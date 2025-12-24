import torch.nn as nn
from torchvision import models


class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=512, pretrained=True):
        super(CRNN, self).__init__()
        efficientnet = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn = efficientnet.features
        self.feature_dim = 2048
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            self.feature_dim,
            hidden_size,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    def forward(self, x):
        conv = self.cnn(x)
        conv = self.adaptive_pool(conv)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        output, _ = self.rnn(conv)
        output = self.fc(output)
        return output
