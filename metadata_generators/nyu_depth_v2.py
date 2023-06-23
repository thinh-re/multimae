import glob
import os
from typing import Dict, List
import json
import sys
import os

sys.path.append(os.getcwd())

########################################################################
train_dir = "nyu_data/data/nyu2_train"
samples: List[Dict[str, str]] = []
for dir_name in os.listdir(train_dir):
    rgbs = glob.glob(os.path.join(train_dir, dir_name, "*.jpg"))
    depths = glob.glob(os.path.join(train_dir, dir_name, "*.png"))

    for rgb in rgbs:
        base_name = os.path.basename(rgb)
        file_name = os.path.splitext(base_name)[0]
        depth = os.path.join(train_dir, dir_name, f"{file_name}.png")
        assert depth in depths, f"File not found {depth}"

        samples.append(dict(rgb=rgb, depth=depth))

################################################################
test_dir = "nyu_data/data/nyu2_test"
rgbs = glob.glob(os.path.join(test_dir, "*_colors.png"))
depths = glob.glob(os.path.join(test_dir, "*_depth.png"))

for rgb in rgbs:
    base_name = os.path.basename(rgb)
    file_name = os.path.splitext(base_name)[0].split("_")[0]
    depth = os.path.join(test_dir, f"{file_name}_depth.png")
    assert depth in depths, f"File not found {depth}"

    samples.append(dict(rgb=rgb, depth=depth))

################################################################
print("Number of samples", len(samples))
json_object = json.dumps(dict(samples=samples), indent=4)

# Writing to sample.json
with open("metadata.json", "w") as outfile:
    outfile.write(json_object)
