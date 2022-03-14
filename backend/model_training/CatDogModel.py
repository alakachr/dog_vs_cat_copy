import torch.nn as nn


class CatDogModel(nn.Module):
    """
    A class used to represent the pytorch model

    Attributes
    ----------
    mobile_net : The model backbone (feature extractor)

    Methods
    -------
    forward(x):
        Apply the model to the input image x
    """

    def __init__(self, mobile_net):
        """
        Parameters
        ----------
        mobile_net : The model backbone (feature extractor)

        """
        super().__init__()

        self.mobile_net = mobile_net

    def forward(self, x):
        """
        Compute model output

        """
        out = self.mobile_net(x)

        return out
