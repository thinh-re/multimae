import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor

from utils.datasets import build_transform


class args:
    input_size = 224
    imagenet_default_mean_and_std = True
    color_jitter = 0.4
    aa = "rand-m9-mstd0.5-inc1"
    train_interpolation = "bicubic"
    reprob = 0.0
    remode = "pixel"
    recount = 1
    resplit = False


transforms = build_transform(is_train=True, args=args)
print(transforms)

img = Image.open("datasets/RGB/1_03-11-09.jpg")


for i in range(10):
    transformed_img: Tensor = transforms(img)
    transformed_img = transformed_img.permute((1, 2, 0)).numpy()
    # print(type(transformed_img), transformed_img.shape, np.max(transformed_img), np.min(transformed_img))

    plt.imshow(transformed_img)
    plt.show()
    plt.savefig(f"tmp/tmp_{i}.jpg")
