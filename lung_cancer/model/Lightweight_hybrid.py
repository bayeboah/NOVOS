import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, causal=False):
        super(DepthwiseConv1D, self).__init__()
        if causal:
            padding = (kernel_size - 1) * dilation  # Adjust padding for dilated causal convolutions
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return f.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseConv1D(channels, channels, kernel_size)
        self.conv2 = DepthwiseConv1D(channels, channels, kernel_size)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.bn(self.pointwise(x + self.conv2(self.conv1(x))))


class EfficientAttention(nn.Module):
    def __init__(self, channels):
        super(EfficientAttention, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        weight = torch.sigmoid(self.fc(self.global_avg(x)))
        weight = torch.softmax(weight, dim=1)  # Improve feature weighting
        return x * weight


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = DepthwiseConv1D(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = DepthwiseConv1D(out_channels, out_channels, kernel_size, dilation=dilation* 2)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        return self.bn(x)


class LightweightModel(nn.Module):
    def __init__(self, input_size, lstm_units=64, tcn_channels=64, num_classes=3):
        super(LightweightModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.lstm = nn.LSTM(input_size, lstm_units, batch_first=True, bidirectional=True, dropout=0.3)
        self.layer_norm = nn.LayerNorm(2 * lstm_units)
        self.tcn = TCNBlock(2 * lstm_units, tcn_channels)

        self.conv_block = DepthwiseConv1D(tcn_channels, tcn_channels)
        self.residual_block = ResidualBlock(tcn_channels)

        self.attention_block = EfficientAttention(tcn_channels * 2)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Sequential(
                  nn.LayerNorm(tcn_channels * 2),
             nn.Linear(tcn_channels * 2, num_classes)
                 )


    def forward(self, x):
        x = self.batch_norm(x)

        x = x.view(x.size(0), x.size(2), -1)  # Reshape to 1D

        x, _ = self.lstm(x)

        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, sequence_length)

        x = self.tcn(x)

        x1 = self.conv_block(x)
        x2 = self.residual_block(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.bn2(x)
        x = self.attention_block(x)
        # x = self.bn2(x)

        x = f.adaptive_avg_pool1d(x, 1).squeeze(2)
        return self.fc(x)