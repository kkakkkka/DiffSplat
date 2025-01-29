from torch.utils.data.dataset import IterableDataset
from torch.utils.data.dataloader import DataLoader


# This is a pseudo class that constructs a dataset from chunks
class ChunkedDataset(IterableDataset):
    raise NotImplementedError("Please implement your own dataloading logic")


# This is a pseudo class that loads data in chunks from HDFS
class ChunkedDataLoader(DataLoader):
    raise NotImplementedError("Please implement your own dataloading logic")
