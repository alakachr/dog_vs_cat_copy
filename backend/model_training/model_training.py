# Code inspired by https://www.kaggle.com/taldanko/pytorch-cats-vs-dogs-cnn
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

import os
import glob
from torchvision import transforms
import torch.nn as nn
from CatDogDataset import CatDogDataset
from torch.utils.data import DataLoader

from CatDogModel import CatDogModel
import config


def train(
    model, num_epochs, train_dataloader, device, criterion, optimizer, val_dataloader
):
    """Launch the training and save the model state dictionnary"""

    train_losses = []
    val_losses = []
    accuracy_list = []
    model.to(device)
    for epoch in range(num_epochs):

        # perform training on train set
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_dataloader):

            # load to gpu
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # backprop and update model params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate training loss for the epoch
        train_losses.append(running_loss / len(train_dataloader))

        # calculate loss accuracy on validation set
        model.eval()
        running_loss = 0
        num_correct = 0
        num_predictions = 0

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader):

                # load to gpu
                images = images.to(device)
                labels = labels.to(device)

                # forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # calculate accuracy for batch
                _, predicted = torch.max(outputs.data, 1)
                num_correct += (predicted == labels).sum().item()
                num_predictions += labels.size(0)

        # calculate val loss for epoch
        val_losses.append(running_loss / len(val_dataloader))

        # calculate accuracy for epoch
        accuracy = num_correct / num_predictions * 100
        accuracy_list.append(accuracy)

        print(
            "[Epoch: %d / %d],  [Train loss: %.4f],  [Test loss: %.4f],  [Acc: %.2f]"
            % (epoch + 1, num_epochs, train_losses[-1], val_losses[-1], accuracy)
        )

    model = model.to("cpu")
    torch.save(model.state_dict(), config.model_path)


def main():

    train_dir = config.train_dataset

    train_images = glob.glob(os.path.join(train_dir, "*.jpg"))

    train_list, val_list = train_test_split(train_images, test_size=0.2)
    # define transformations for the train, test and holdout images
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.mean_image_norm,
                std=config.std_image_norm,
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # create dataset objects
    train_dataset = CatDogDataset(train_list, transform=train_transforms)
    val_dataset = CatDogDataset(val_list, mode="val", transform=val_transforms)

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    num_epochs = config.num_epochs
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("DEVICE", device)

    model_backbone = torch.hub.load(
        "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
    )
    model_backbone.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    model = CatDogModel(model_backbone)

    # applies log_softmax and then NLLLoss cost function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train(
        model,
        num_epochs,
        train_dataloader,
        device,
        criterion,
        optimizer,
        val_dataloader,
    )


if __name__ == "__main__":
    main()
