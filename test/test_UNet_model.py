import torch
import torch.nn as nn
import sys
sys.path.append('src')
from UNet_model import conv_block
from UNet_model import encoder_block
from UNet_model import decoder_block
from UNet_model import build_unet
import pytest

@pytest.fixture

def my_unet_model():
    return nn.Module.conv_block
    return nn.Module.encoder_block
    return nn.Module.decoder_block
    return nn.Module.build_unet

def test_imports():
    assert True == True
