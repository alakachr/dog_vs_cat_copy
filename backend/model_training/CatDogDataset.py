from PIL import Image
from torch.utils.data import Dataset

# make a dataset class
class CatDogDataset(Dataset):
    """
    A class used to represent the dataset

    Attributes
    ----------
    images_list : list of the images filenames )
    mode: train or val mode
    transform: image preprocessing function


    """

    def __init__(self, images_list, mode="train", transform=None):
        """
        Parameters
        ----------
        images_list : list of the images filenames )
        mode: train or val mode
        transform: image preprocessing function
        """
        self.images_list = images_list
        self.mode = mode
        self.transform = transform

    # dataset length
    def __len__(self):
        """
        Return Dataset Len
        """
        self.dataset_len = len(self.images_list)
        return self.dataset_len

    # load an image
    def __getitem__(self, idx):
        """
        Returns a Dataset element
        """
        image_name = self.images_list[idx]
        image = Image.open(image_name)
        image = image.resize(
            (256, 256)
        )  # this is important when feeding into a pretrained model
        transformed_image = self.transform(image)
        image_category = image_name.split("/")[-1].split(".")[0]

        if self.mode == "train" or self.mode == "val":
            if image_category == "cat":
                label = 0
            else:
                label = 1
            return transformed_image, label
        else:
            return transformed_image
