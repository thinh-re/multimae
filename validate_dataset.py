import os, json
from typing import List
from PIL import Image
import numpy as np
from tqdm import tqdm

# dataset_names = ["nyu-depth-v2", "multimae-v1"]
dataset_names = ["nyu-depth-v2"]

for dataset_name in dataset_names:
    print("Process dataset", dataset_name)
    with open(os.path.join("datasets_metadata", f"{dataset_name}.json"), "r") as f:
        obj = json.load(f)

    max_lst: List[int] = []
    min_lst: List[int] = []
    for sample in tqdm(obj["samples"]):
        depth_path = os.path.join("datasets", dataset_name, sample["depth"])
        if "nyu2_train" in depth_path:
            depth = Image.open(depth_path)
            depth = np.array(depth)

            max_lst.append(np.max(depth))
            min_lst.append(np.min(depth))

    print("max", np.max(max_lst), np.mean(max_lst))
    print("min", np.min(min_lst), np.mean(min_lst))
