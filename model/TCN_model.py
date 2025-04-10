import torch
import torch.nn as nn
import torch.nn.functional as F
class MaxNorm(nn.Module):
    def __init__ (self,eps=1e-5):
        """
        MaxNorm Layer for normalization.

        Parameters:
        eps (float): Small value to avoid division by zero.
        """
        super(MaxNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
         x: [batch_size, num_channels, num_frames]
         Calculate the maximum value across the channels

        """
        m,_ = torch.max(x, dim=1, keepdim=True)
        return x / (m + self.eps)

class ResidualBlock_encoder(nn.Module):
    def __init__ (self, in_channels, out_channels, kernel_size=3,drop_out=0.2):
        """
        Residual Block for TCN.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        residual (bool): Whether to use residual connections.
        """
        super(ResidualBlock_encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop_out)
        self.max_pool= nn.MaxPool1d(kernel_size=2, stride=2)
        self.norm = MaxNorm(eps=1e-5)

        # residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv =  nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        residual = self.max_pool(residual)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.max_pool(x)
        x = x + residual
        x = self.norm(x)
        return x


class TCN_encoder(nn.Module):
    def __init__(self,input_channels, num_channels= [32, 64, 96], kernel_size = 3,dropout_rate = 0.2):

        super(TCN_encoder, self).__init__()
        self.num_channels = num_channels
        self.encoder_layers = nn.ModuleList()

        # First layer
        self.encoder_layers.append(
            ResidualBlock_encoder(input_channels, num_channels[0], kernel_size, dropout_rate)
        )

        # Remaining layers
        for i in range(1, len(num_channels)):
            self.encoder_layers.append(
                ResidualBlock_encoder(num_channels[i - 1], num_channels[i], kernel_size, dropout_rate)
            )

    def forward(self, x):

        encoder_features = []
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_features.append(x)

        return x, encoder_features

class RepeatUpsampling(nn.Module):
    def __init__(self):
        super(RepeatUpsampling,self).__init__()
    def forward(self,x):
        batch_size, channels, time = x.size()
        # Repeat each frame 2 times
        x = x.unsqueeze(3)
        x = x.repeat(1, 1, 1, 2)
        x = x.view(batch_size, channels, time * 2)
        return x

class ResidualBlock_decoder(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=3, dropout_rate=0.2):
        super(ResidualBlock_decoder, self).__init__()

        self.upsample = RepeatUpsampling()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = MaxNorm(eps=1e-5)
        # residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = self.upsample(x)
        residual = self.residual_conv(residual)

        x = self.upsample(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x


class TCN_decoder(nn.Module):
    def __init__(self, output_channels, kernel_size=3, dropout_rate=0.2, num_channels=[96, 64, 32]):
        super(TCN_decoder,self).__init__()
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.decoder_layers = nn.ModuleList()

        # First layer
        self.decoder_layers.append(
            ResidualBlock_decoder(num_channels[0], num_channels[1], kernel_size, dropout_rate)
        )

        # Middle decoder layer
        self.decoder_layers.append(
            ResidualBlock_decoder(num_channels[1], num_channels[2], kernel_size, dropout_rate)
        )

        self.final_layer = nn.Sequential(
            RepeatUpsampling(),
            nn.Conv1d(num_channels[2], output_channels, kernel_size, padding=(kernel_size - 1) // 2),
        )

        self.classifier = nn.Linear(output_channels, output_channels)


    def forward(self, x):
        decoder_features = []
        for layer in self.decoder_layers:
            x = layer(x)
            decoder_features.append(x)
        x = self.final_layer(x)
        batch,channels,time = x.size()
        x = x.permute(0,2,1)

        x = self.classifier(x)

        return x, decoder_features

class TCN(nn.Module):
    def __init__(self,input_channels,output_channels, num_classes, num_channels = [32, 64, 96], kernel_size = 3, dropout_rate = 0.2):
        super(TCN,self).__init__()
        self.encoder = TCN_encoder(input_channels, num_channels, kernel_size, dropout_rate)
        self.decoder = TCN_decoder(output_channels, kernel_size, dropout_rate)
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x, encoder_features = self.encoder(x)
        x, decoder_features = self.decoder(x)
        logits = self.fc(x)

        # since the video have same class over time, we can use the last frame or average over time
        # we choos mean
        # logits = torch.mean(logits, dim=1)

        return logits,encoder_features, decoder_features















