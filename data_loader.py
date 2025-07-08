import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from vocabulary import Vocabulary


class FashionProductDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, 
                 freq_threshold=5):
        self.root_dir = root_dir
        # Handle CSV parsing errors with proper parameters
        try:
            self.df = pd.read_csv(caption_file, on_bad_lines='skip',
                                  engine='python')
        except Exception as e:
            # Fallback: try with different separator or error handling
            self.df = pd.read_csv(caption_file, sep=',', on_bad_lines='skip',
                                  quoting=1, engine='python')
        
        self.transform = transform

        # Filter out rows where productDisplayName is NaN
        self.df = self.df.dropna(subset=['productDisplayName'])
        
        # Get image IDs and product names from the dataframe
        self.img_ids = self.df["id"]
        self.captions = self.df["productDisplayName"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions.iloc[idx]
        img_id = self.img_ids.iloc[idx]
        img_name = f"{img_id}.jpg"
        img_location = os.path.join(self.root_dir, img_name)
        
        # Handle missing images gracefully
        try:
            img = Image.open(img_location).convert("RGB")
        except (FileNotFoundError, OSError):
            # Create a placeholder image if the actual image is missing
            img = Image.new('RGB', (224, 224), color='gray')

        # Apply the transformation to the image
        if self.transform is not None:
            img = self.transform(img)

        # Numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)