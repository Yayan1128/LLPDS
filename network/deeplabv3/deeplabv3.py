from .aspp import *
from functools import partial
# from utils1.afnb2 import PMR
from .adr import ADRBlock

class DeepLabv3Plus(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=2, output_dim=256,alpha=0.6):
        super(DeepLabv3Plus, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            # aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            # aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4
        self.alpha=alpha

        # self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

   
        self.deconv2 = nn.ConvTranspose2d(in_channels=256,
                                          out_channels=256,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.deconv1 = nn.ConvTranspose2d(in_channels=2048,
                                          out_channels=256,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )


        self.deconv3 = nn.ConvTranspose2d(in_channels=304,
                                          out_channels=304,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.deconv4 = nn.ConvTranspose2d(in_channels=304,
                                          out_channels=304,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.bn1 = nn.BatchNorm2d(num_features=256, )
        self.relu = nn.ReLU(inplace=True)

        self.bn3 = nn.BatchNorm2d(num_features=304, )
        self.b=ADRBlock(2048,alpha=self.alpha)
        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )
        self.representation1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1)
        )
    

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
    
    def decoder(self,features):
        x_low=features[0]
        feature1e_m=features[1]
        feature1e_le=features[2]
        feature1e_le=self.b(feature1e_le,feature1e_m)
        x_low = self.project(x_low)#64
  
        feature = self.deconv1(feature1e_le)
        feature = self.bn1(feature)
        feature = self.relu(feature)


        feature = self.deconv2(feature)
        feature = self.bn1(feature)
        output_feature = self.relu(feature)
        output_feature= F.interpolate(output_feature, size=x_low.shape[2:], mode='nearest')
     
        # output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='nearest', align_corners=True)
        pre_out=torch.cat([x_low, output_feature], dim=1)


        pre_out = self.deconv3(pre_out)
        pre_out = self.bn3(pre_out)
        pre_out = self.relu(pre_out)

        pre_out = self.deconv4(pre_out)
        pre_out = self.bn3(pre_out)
        pre_out = self.relu(pre_out)

        prediction = self.classifier(pre_out)
        representation = self.representation(pre_out)
        representation1 = self.representation1(output_feature)
        # representation2 = self.representation2(output_feature)
        representations=[representation,representation1]
        return prediction, representations
       
        return prediction
        
    def encoder(self,x):
        x_2 = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x_2)#64
        # print(x.size())

        x_low = self.resnet_layer1(x)#64
        # print(x_low.size())
        x_8 = self.resnet_layer2(x_low)#32
        # print(x.size())
        x_16 = self.resnet_layer3(x_8)#16
        # print(x.size())
        feature1e_le = self.resnet_layer4(x_16)#16  
        return [x_low,x_16,feature1e_le]

    def forward(self, x,tok_lern=False):
     
      
        features=self.encoder(x)

        prediction, representations=self.decoder(features)
        if tok_lern:
          return prediction,representations, features
        else:
          return prediction,representations

