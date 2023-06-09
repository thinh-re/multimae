from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(
    Image.open("datasets/nyu-depth-v2/nyu_data/data/nyu2_test/00000_depth.png")
)
img = img / (2**16) * (2**8)
print(img.shape)
print(np.min(img), np.max(img))
b, bins, patches = plt.hist(img.ravel(), 255)
plt.xlim([0, 255])
plt.show()
plt.savefig("tmp/hehe.png")
