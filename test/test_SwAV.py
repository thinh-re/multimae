import torch
import glob
from PIL import Image
import torchvision.transforms.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torch.hub.load("facebookresearch/swav:main", "resnet50")
model.eval()
model.to(device)
# print(model.__dict__)

image_paths = glob.glob("datasets/RGB/*.jpg")

for image_path in image_paths:
    image = Image.open(image_path)
    image = F.to_tensor(image).unsqueeze(0)
    image = image.to(device)
    embedding = model(image)
    print(image_path, embedding.shape)
