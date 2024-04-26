import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel
from transformers import BigBirdModel, BigBirdConfig


class CustomAlbert(BigBirdModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir):

    model = BigBirdModel.from_pretrained("google/bigbird-roberta-large")

  

    return model