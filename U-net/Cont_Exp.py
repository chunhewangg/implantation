import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def conv_block(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

class Encoder(nn.Module):
    def __init__(self, input_dim =1, channel1 = 64, channel2 = 128, channel3 = 256, channel4 = 512, channel5 = 1024):
        super(Encoder,self).__init__()
        # input dimension
        self.input_dim = input_dim

        # channels
        self.out_channel1 = channel1
        self.out_channel2 = channel2
        self.out_channel3 = channel3
        self.out_channel4 = channel4
        self.out_channel5 = channel5

        self.block1 = conv_block(self.input_dim, self.out_channel1)
        self.block2 = conv_block(self.out_channel1, self.out_channel2)
        self.block3 = conv_block(self.out_channel2, self.out_channel3)
        self.block4 = conv_block(self.out_channel3, self.out_channel4)
        self.block5 = conv_block(self.out_channel4, self.out_channel5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        x1 = self.block1(x)
        x = self.pool(x1)

        x2 = self.block2(x)
        x = self.pool(x2)

        x3 = self.block3(x)
        x = self.pool(x3)

        x4 = self.block4(x)
        x = self.pool(x4)

        x5 = self.block5(x)
        return x1, x2, x3, x4, x5


class Decoder(nn.Module):
        def __init__(self, input_dim =1024):
            super(Decoder,self).__init__()

            self.input_dim = input_dim

            self.block1 = conv_block(self.input_dim+512, 512)
            self.block2 = conv_block(512+256, 256)
            self.block3 = conv_block(256+128, 128)
            self.block4 = conv_block(128+64, 64)
            self.block5 = nn.Conv2d(64, 1, kernel_size=1)
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(512, 512, kernel_size=3, padding=1)
            )
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
            )
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )

        def forward(self, x1, x2, x3, x4, x5):
            x = self.up1(x5)
            x = torch.cat((x4,x), dim=1)
            x = self.block1(x)
            x = self.up2(x)
            x = torch.cat((x3, x), dim=1)
            x = self.block2(x)
            x = self.up3(x)
            x = torch.cat((x2, x), dim=1)
            x = self.block3(x)
            x = self.up4(x)
            x = torch.cat((x1, x), dim=1)
            x = self.block4(x)
            x = self.block5(x)
            return x
        
class UNet(nn.Module):
    def __init__(self, input_dim =1):
        super(UNet,self).__init__()
        self.encoder = Encoder(input_dim=input_dim)
        self.decoder = Decoder(input_dim=1024)

    def forward(self,x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4, x5)
        return out



if __name__ == "__main__":
    model = UNet(input_dim=1)
    x = torch.randn(1, 1, 128, 128)  # batch size 1, 1 channel, 128x128 image
    out = model(x)
    print("Output shape:", out.shape) 



       
        
