from typing import List, Tuple
import torch
from PIL import Image
import open_clip
from torch import Tensor
import glob
from sklearn.manifold import TSNE
import wandb
from pprint import pprint

images_path = glob.glob("images/01_raw/kd_dataset/*/*")
pprint(images_path)
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="laion400m_e32"
)
images = torch.concat(
    [preprocess(Image.open(image_path)).unsqueeze(0) for image_path in images_path]
)

text = open_clip.tokenize(
    ["a girl standing near a bridge", "a girl", "a bridge", "a cat"]
)

tsne = TSNE(
    n_components=2,
    learning_rate="auto",
    init="random",
    metric="cosine",
    random_state=10,
    n_jobs=-1,  # using all processors
)

with torch.no_grad():
    wandb.login(key="21164645663e517404292c3a0f69a9ae57220ed6")
    wandb_run = wandb.init(
        name="open_clip",
        project="ArcFace",
        resume="auto",
        id="open_clip",
    )

    image_features: Tensor = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    print(image_features.numpy().shape)

    embeddings_in_2D = tsne.fit_transform(image_features.numpy()).tolist()

    data: List[Tuple[wandb.Image, float, float, str]] = [
        # "wandb.Image(...)", "dim_1", "dim_2", "label"
    ]
    for i, (image_path, embedding) in enumerate(zip(images_path, embeddings_in_2D)):
        label = image_path.split("/")[-2]
        data.append(
            [
                i,  # wandb.Image(cv2.resize(np.transpose(img, (1,2,0)), (50, 50))),
                embedding[0],
                embedding[1],
                label,
            ]
        )

    columns: List[str] = ["image", "dim_1", "dim_2", "label"]

    wandb_run.log(
        {
            f"tsne": wandb.Table(data=data, columns=columns),
        }
    )

    # text_probs = (100.0 * image_features @ image_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

# image = preprocess(Image.open("COME_Train_84.jpg")).unsqueeze(0)
# text = open_clip.tokenize(["a girl standing near a bridge", "a girl", "a bridge", "a cat"])

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
