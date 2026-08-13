"""ImageNet-1K subset loaders for the paper CNN experiments."""
import os

import torch
from torchvision import datasets, transforms


class IncrementalImageNet1k:
    """ImageNet-1K N-class subset for incremental learning.

    Expects the layout produced by ``cnn/tools/process_imagenet.py``::

        <root>/train/<wnid>/*.JPEG
        <root>/val/<wnid>/*.JPEG
        <root>/test/<wnid>/*.JPEG

    Args:
        root: directory containing ``train/``, ``val/``, ``test/`` folders.
        num_classes: use only the first ``num_classes`` wnids (ImageFolder
            alphabetical ordering, same as the processing script).
        resize: input resolution fed to the model (224 by default).
    """

    def __init__(
        self,
        root: str = "data/imagenet1k-100-processed",
        num_classes: int = 100,
        resize: int = 224,
    ):
        n_on_disk = len(os.listdir(os.path.join(root, "train")))
        if not (1 <= num_classes <= n_on_disk):
            raise ValueError(
                f"num_classes must be in [1, {n_on_disk}], got {num_classes}"
            )
        self.num_classes = num_classes

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(int(round(resize * 256 / 224))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

        full_train = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        full_val = datasets.ImageFolder(os.path.join(root, "val"), transform=eval_tf)
        full_test = datasets.ImageFolder(os.path.join(root, "test"), transform=eval_tf)

        if num_classes < n_on_disk:
            self.train_set = torch.utils.data.Subset(
                full_train,
                [i for i, t in enumerate(full_train.targets) if t < num_classes],
            )
            self.val_set = torch.utils.data.Subset(
                full_val,
                [i for i, t in enumerate(full_val.targets) if t < num_classes],
            )
            self.test_set = torch.utils.data.Subset(
                full_test,
                [i for i, t in enumerate(full_test.targets) if t < num_classes],
            )
        else:
            self.train_set = full_train
            self.val_set = full_val
            self.test_set = full_test

        self._class_indices = {"train": {}, "test": {}, "val": {}}
        self._precompute_class_indices()
        self._cache_indices = {"train": {}, "test": {}, "val": {}}

    def _precompute_class_indices(self):
        for mode, dataset in [
            ("train", self.train_set),
            ("test", self.test_set),
            ("val", self.val_set),
        ]:
            if hasattr(dataset, "targets"):
                targets = torch.tensor(dataset.targets, dtype=torch.long)
            else:
                base_targets = dataset.dataset.targets
                targets = torch.tensor(
                    [base_targets[i] for i in dataset.indices], dtype=torch.long
                )
            class_to_indices = {}
            for lbl in range(self.num_classes):
                mask = targets == lbl
                if mask.any():
                    class_to_indices[lbl] = mask.nonzero(as_tuple=True)[0]
            self._class_indices[mode] = class_to_indices

    def get_set(self, mode, label):
        if mode == "train":
            data = self.train_set
        elif mode == "test":
            data = self.test_set
        elif mode == "val":
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train', 'test', or 'val'")

        try:
            label_key = tuple(sorted(int(x) for x in label))
        except TypeError:
            label_key = (int(label),)

        cache = self._cache_indices[mode]
        if label_key in cache:
            return cache[label_key]

        class_indices = self._class_indices[mode]
        indices_list = [class_indices[lbl] for lbl in label_key if lbl in class_indices]
        indices = torch.cat(indices_list).tolist() if indices_list else []
        subset = torch.utils.data.Subset(data, indices)
        cache[label_key] = subset
        return subset

    def get_loader(self, mode, label, batch_size=64, shuffle=None):
        subset = self.get_set(mode, label)
        shuffle = mode == "train"

        use_cuda = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 2
        num_workers = int(os.environ.get("_DATALOADER_NUM_WORKERS", 0)) or max(
            2, min(8, cpu_count // 2)
        )

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": use_cuda,
        }
        if num_workers > 0:
            loader_kwargs.update({
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2,
            })
        return torch.utils.data.DataLoader(subset, **loader_kwargs)


DATASET_CHOICES = ("imagenet1k",)


def build_dataset(name: str, **kwargs):
    """Instantiate the paper ImageNet-1k incremental dataset manager."""
    name = name.lower()
    if name == "imagenet1k":
        return IncrementalImageNet1k(
            root=kwargs.get("root", "data/imagenet1k-100-processed"),
            num_classes=kwargs.get("num_classes", 100),
            resize=kwargs.get("img_size", 224),
        )
    raise ValueError(f"Unknown dataset '{name}'. Valid: {DATASET_CHOICES}")
