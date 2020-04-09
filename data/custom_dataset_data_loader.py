import torch.utils.data
from data.base_data_loader import BaseDataLoader
from pdb import set_trace as st


def CreateDataset(opt):
    dataset = None
    if opt.fgnet:
        from data.fgnet_dataset import FGNET_Dataset
        dataset = FGNET_Dataset()
    elif opt.youtube:
        from data.youtube_dataset import Youtube_Dataset
        dataset = Youtube_Dataset()
    elif 'ffhq_aging_new_labels_our_alignment' not in opt.dataroot:
        from data.multiclass_unaligned_dataset_old import MulticlassUnalignedDataset
        dataset = MulticlassUnalignedDataset()
    else:
        from data.multiclass_unaligned_dataset import MulticlassUnalignedDataset
        dataset = MulticlassUnalignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            drop_last=True,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
