import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch



def collate_fn(batch):

    x1, y1, x2, y2, label, weight_= list(zip(*batch))
    offset, count = [], 0
    for item in x1:
        count += item.shape[0]
        offset.append(count)
    for i, item in enumerate(label):
        item = item.view(1, 256, 256)
        item = torch.unsqueeze(item, dim=0)
        if i == 0:
            target = item
        else:
            target = torch.cat((target,item),dim=0)

    for i, item in enumerate(weight_):
        item = item.view(1, 256, 256)
        item = torch.unsqueeze(item, dim=0)
        if i == 0:
            weight = item
        else:
            weight = torch.cat((weight,item),dim=0)


    return torch.cat(x1), torch.cat(y1), torch.cat(x2), torch.cat(y2), torch.IntTensor(offset), target, weight

class DInterface(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.num_workers
        self.dataset = cfg.dataset
        self.batch_size = cfg.batch_size
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(split='train')
            self.valset = self.instancialize(split='val')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(split='test')



    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)


    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)

    def load_data_module(self):
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        return self.data_module(other_args, self.cfg.data_root, self.cfg.label_root)
