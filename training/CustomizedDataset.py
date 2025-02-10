import pickle
import torch
from torch.utils.data import Dataset

class CustomizedDataset(Dataset):
    def __init__(self, metadata):
        super(CustomizedDataset, self).__init__()
        self.data = pickle.load(open(metadata, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with torch.no_grad():
            sample = self.data[idx]
            text_embed = sample.get('text_embed', None)
            audio_embed = sample.get('audio_embed', None)
            label = sample.get('label', None)

            # Debugging output
            if text_embed is None or audio_embed is None or label is None:
                print(f"Warning: None value found at index {idx}")
                return None  # Returning None will be handled by a custom collate function

            return text_embed, audio_embed, label
