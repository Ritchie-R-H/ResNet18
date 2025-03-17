import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch

class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  
        y = self.conv(y.unsqueeze(1)).squeeze(1)  
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)  
        return x * y  

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batchsize, channel, 32, 32) ->(batchsize, channel, 1, 1)
        out = self.squeeze(x)
        # (batchsize, channel, 1, 1) -> (batchsize, channel)
        out = out.view(out.size(0), -1)
        out = self.excitation(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        return x * out

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, r=16, p=0.2):
        super().__init__()
        self.should_downsample = (in_channels != out_channels) or (stride != 1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # downsample layer
        if self.should_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        # SE block
        self.se = SEBlock(out_channels, r)

        self.stochastic_depth = nn.Dropout2d(p)

    def forward(self, x):
        identity = x
        
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.dropout(self.conv2(out)))
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)
    

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()

        # Initial Convolution (C1 = 32 channels)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual layers with Max Pooling after each layer
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)  # Max pooling after layer1

        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)  # Max pooling after layer2

        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)  # Max pooling after layer3

        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Average Pooling (Final layer)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 800),  # First FC layer
            nn.ReLU(),
            nn.Linear(800, num_classes)  # Output layer
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(SEResidualBlock(in_channels, out_channels, stride=stride))
        layers.append(SEResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.pool1(x)  # Apply max pooling after layer1

        x = self.layer2(x)
        x = self.pool2(x)  # Apply max pooling after layer2

        x = self.layer3(x)
        x = self.pool3(x)  # Apply max pooling after layer3

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = CustomResNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (3, 32, 32))