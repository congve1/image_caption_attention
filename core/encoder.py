import torch
import torch.nn as nn
import torchvision.models as models
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-3]) #get conv4 features
    
    def forward(self, imgs):
        features = self.resnet(imgs)
        features = features.reshape(*(features.size()[:-2]+(-1,)))
        return features #BxDxL