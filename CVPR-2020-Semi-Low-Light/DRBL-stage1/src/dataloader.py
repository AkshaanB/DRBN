import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class MSDataLoader(DataLoader):
    """
    A DataLoader wrapper that appends `idx_scale` to every batch.

    The original implementation subclassed the private `_DataLoaderIter`
    class (removed in PyTorch >= 1.2) to inject a scale index into each
    batch.  This version achieves the same result by wrapping the standard
    DataLoader iterator in a generator, keeping all code compatible with
    modern PyTorch without relying on any private API.
    """

    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=default_collate, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):

        super(MSDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=args.n_threads,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

        self.scale = args.scale

    def __iter__(self):
        """Yield batches with idx_scale appended as the last element."""
        for batch in super().__iter__():
            idx_scale = 0
            if len(self.scale) > 1 and self.dataset.train:
                idx_scale = random.randrange(0, len(self.scale))
                self.dataset.set_scale(idx_scale)

            # default_collate returns a list; append idx_scale to it
            yield list(batch) + [idx_scale]
