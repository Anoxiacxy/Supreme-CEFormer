from typing import Union, Optional, Any

from pl_bolts.datamodules import CIFAR10DataModule
from torchvision.datasets import CIFAR100


class CIFAR100DataModule(CIFAR10DataModule):

    name = "cifar100"
    dataset_cls = CIFAR100

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )

    @property
    def num_classes(self) -> int:
        """
        :return: 100
        """
        return 100