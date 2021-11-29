from UNet_model import *
from torchsummary import summary

inputs = torch.randn((2, 3, 256, 256))
model = build_unet()
summary(model, (2, 3, 256, 256))
