import os, json
from typing import List
from PIL import Image
import numpy as np
from tqdm import tqdm

"""
Preprocessing:

- Normalize depth in range [0,255]
"""

dataset_names = ["nyu-depth-v2", "multimae-v1"]

for dataset_name in dataset_names:
    print("Process dataset", dataset_name)
    with open(os.path.join("datasets_metadata", f"{dataset_name}.json"), "r") as f:
        obj = json.load(f)

    def normalize(x: np.array) -> np.array:
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    max_lst: List[int] = []
    min_lst: List[int] = []
    for sample in tqdm(obj["samples"]):
        depth_path = os.path.join("datasets", dataset_name, sample["depth"])
        depth = Image.open(depth_path)
        depth = np.array(depth)

        if depth.dtype == np.int32:
            depth = depth / 2**8
            depth = depth.astype(np.uint8)
            depth = Image.fromarray(depth)
            print(depth_path)
            depth.save(depth_path)

    #     max_lst.append(np.max(depth))
    #     min_lst.append(np.min(depth))

    # print("max", np.max(max_lst))
    # print("min", np.min(min_lst))
