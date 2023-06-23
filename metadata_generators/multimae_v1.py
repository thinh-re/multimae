import os, glob
import sys
import os

sys.path.append(os.getcwd())

rgbs = glob.glob(os.path.join("rgb", "sod", "*"))
rgb_exts = list(set([rgb.split(".")[-1] for rgb in rgbs]))
print("rgb_exts", rgb_exts)  # ['jpg']

depths = glob.glob(os.path.join("depth", "sod", "*"))
depth_exts = list(set([depth.split(".")[-1] for depth in depths]))
print("depth_exts", depth_exts)  # ['png']

rgbs = glob.glob(os.path.join("rgb", "sod", "*.jpg"))
depths = glob.glob(os.path.join("depth", "sod", "*.png"))

samples = []

for rgb in rgbs:
    base_name = os.path.basename(rgb)
    file_name = os.path.splitext(base_name)[0]
    depth = os.path.join("depth", "sod", f"{file_name}.png")
    assert depth in depths, f"File not found {depth}"
    samples.append(dict(rgb=rgb, depth=depth))

print("Num samples", len(samples))

import json

json_object = json.dumps(dict(samples=samples), indent=4)

# Writing to sample.json
with open("metadata.json", "w") as outfile:
    outfile.write(json_object)
