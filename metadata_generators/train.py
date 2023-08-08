import json
from typing import Dict
import os
import random
import sys
import os

sys.path.append(os.getcwd())


def read_json(json_path: str) -> Dict:
    # Opening JSON file
    with open(json_path, "r") as openfile:
        # Reading from json file
        return json.load(openfile)


datasets = ["multimae-v1", "nyu-depth-v2", "imagenet1k"]
# datasets = ["multimae-v1"]
train_ratios = [9, 9, 9.8]
dev_ratios = [0.5, 0.5, 0.1]
test_ratios = [0.5, 0.5, 0.1]

rs = dict(train=dict(samples=[]), validation=dict(samples=[]), test=dict(samples=[]))

for dataset, train_ratio, dev_ratio, test_ratio in zip(
    datasets, train_ratios, dev_ratios, test_ratios
):
    d = read_json(os.path.join("datasets_metadata", f"{dataset}.json"))
    samples: Dict[str, str] = d["samples"]
    random.shuffle(samples)

    for sample in samples:
        if not sample["rgb"].startswith("/"):
            sample["rgb"] = os.path.join(dataset, sample["rgb"])
        if not sample["depth"].startswith("/"):
            sample["depth"] = os.path.join(dataset, sample["depth"])

    total_ratio = train_ratio + dev_ratio + test_ratio
    num_trains = int(len(samples) / total_ratio * train_ratio)
    num_devs = int(len(samples) / total_ratio * dev_ratio)
    num_test = len(samples) - num_trains - num_devs

    rs["train"]["samples"].extend(samples[:num_trains])
    rs["train"]["length"] = len(rs["train"]["samples"])
    rs["validation"]["samples"].extend(samples[num_trains : num_trains + num_devs])
    rs["validation"]["length"] = len(rs["validation"]["samples"])
    rs["test"]["samples"].extend(samples[num_trains + num_devs :])
    rs["test"]["length"] = len(rs["test"]["samples"])

print("Num train:", len(rs["train"]["samples"]))
print("Num validation:", len(rs["validation"]["samples"]))
print("Num test:", len(rs["test"]["samples"]))

# Serializing json
json_object = json.dumps(rs, indent=4)

# Writing to sample.json
with open("datasets_metadata/v1.json", "w") as f:
    f.write(json_object)
