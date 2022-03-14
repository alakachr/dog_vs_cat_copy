# code inspired by https://testdriven.io/blog/fastapi-streamlit/
import config
import numpy as np
import torch
from torchvision import transforms

from model_training.CatDogModel import CatDogModel
import torch.nn as nn

val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def inference(image):
    """Load model and compute model prediction

    Args:
        image:image as an array

    Returns:
        label: the label predicted
    """

    model_backbone = torch.hub.load(
        "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
    )
    model_backbone.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    model = CatDogModel(model_backbone)
    model.load_state_dict(torch.load(config.MODEL_PATH))

    image = image.resize((256, 256))
    image = val_transforms(image)

    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))

    biggest_pred_index = np.array(outputs)[0].argmax()

    label = config.LABELS_DICT[biggest_pred_index]
    return label
