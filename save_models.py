from transformers import pipeline
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
seg_model = deeplabv3_resnet50(pretrained=True)
torch.save(seg_model,'segmentation.pth')
depth_model= torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
torch.save(depth_model,'depth_map.pth')

detect_model=pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
detect_model.save_pretrained("describe.h5")