from UNet_model import *
from torchsummary import summary

model = build_unet()
summary(model, (3, 256, 256))
