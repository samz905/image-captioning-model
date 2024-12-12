import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions, vocab, transform=None):
        self.root_dir = root_dir
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        self.imgs = list(captions.keys())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["< SOS >"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)


class CapsCollate:
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, 
            batch_first=self.batch_first,
            padding_value=self.pad_idx
        )
        return imgs, targets


def get_data_loader(dataset, vocab, batch_size, shuffle=False, num_workers=1):
    pad_idx = vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx, batch_first=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return data_loader
