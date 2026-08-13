import torch
import random
import os
from typing import Dict, Any, Optional
from src.methods.base import BaseContinualMethod


class ReplayMethod(BaseContinualMethod):
    """Experience Replay with iCaRL-style per-class exemplar memory."""

    def __init__(self, *args, memory_per_class: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_per_class = int(memory_per_class)
        self.memory_set = {"data": [], "labels": []}

    def get_training_params(self) -> Dict[str, Any]:
        params = super().get_training_params()
        params["memory_per_class"] = self.memory_per_class
        return params

    def _print_task_info(self, task_idx: int) -> None:
        print(f"Memory size: {len(self.memory_set['data'])} samples")

    def get_active_classes_range(self, task_idx: int) -> Optional[tuple]:
        """Active range spans all seen classes (TIL)."""
        end_class = (task_idx + 1) * self.increment
        return (0, end_class)

    def get_train_loader(self, task_idx: int, train_loader):
        """Iterate over current-task data combined with the replay buffer."""
        start_class = task_idx * self.increment
        end_class = start_class + self.increment
        train_set = self.experiment_dataset.get_set(
            mode="train", label=range(start_class, end_class),
        )
        return self._create_combined_loader(train_set)

    def after_task(self, task_idx: int, train_loader) -> None:
        """Update memory after training each task."""
        start_class = task_idx * self.increment
        end_class = start_class + self.increment
        train_set = self.experiment_dataset.get_set(
            mode="train", label=range(start_class, end_class),
        )
        self._update_memory(train_set)

    def _create_combined_loader(self, train_set):
        """Create a DataLoader combining current task data with memory."""
        use_cuda = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 2
        num_workers = int(os.environ.get("_DATALOADER_NUM_WORKERS", 0)) or max(2, min(8, cpu_count // 2))
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "pin_memory": use_cuda,
        }
        if num_workers > 0:
            loader_kwargs.update({
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2,
            })

        if len(self.memory_set["data"]) == 0:
            print("No memory samples available, training only on current task data")
            return torch.utils.data.DataLoader(train_set, **loader_kwargs)

        try:
            memory_data_tensor = torch.stack(self.memory_set["data"])
            memory_labels_tensor = torch.tensor(self.memory_set["labels"], dtype=torch.long)

            class MemoryDataset(torch.utils.data.Dataset):
                def __init__(self, data_tensor, labels_tensor):
                    self.data = data_tensor
                    self.labels = labels_tensor

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx], self.labels[idx].item()

            memory_dataset = MemoryDataset(memory_data_tensor, memory_labels_tensor)
            combined_dataset = torch.utils.data.ConcatDataset([train_set, memory_dataset])
            return torch.utils.data.DataLoader(combined_dataset, **loader_kwargs)
        except Exception as e:
            print(f"Error creating memory dataset: {e}")
            return torch.utils.data.DataLoader(train_set, **loader_kwargs)

    def _update_memory(self, train_set):
        """Append per-class exemplars from the current task (older ones untouched)."""
        class_to_indices = {}
        targets = None

        # Walk nested Subsets down to the base dataset's targets.
        try:
            ds = train_set
            sub_indices = None
            while isinstance(ds, torch.utils.data.Subset):
                layer_idx = ds.indices
                if torch.is_tensor(layer_idx):
                    layer_idx = layer_idx.tolist()
                if sub_indices is None:
                    sub_indices = list(layer_idx)
                else:
                    sub_indices = [layer_idx[i] for i in sub_indices]
                ds = ds.dataset
            if hasattr(ds, "targets") and sub_indices is not None:
                full_targets = ds.targets
                if torch.is_tensor(full_targets):
                    targets = full_targets[sub_indices].tolist()
                else:
                    targets = [int(full_targets[i]) for i in sub_indices]
        except Exception:
            targets = None

        if targets is not None:
            for i, label in enumerate(targets):
                class_to_indices.setdefault(label, []).append(i)
        else:
            for i, (_, label) in enumerate(train_set):
                class_to_indices.setdefault(int(label), []).append(i)

        samples_per_class = self.memory_per_class
        new_data, new_labels = [], []
        for class_label, indices in class_to_indices.items():
            num_samples = min(samples_per_class, len(indices))
            for idx in random.sample(indices, num_samples):
                data, label = train_set[idx]
                if isinstance(data, torch.Tensor):
                    new_data.append(data.cpu())
                else:
                    new_data.append(torch.tensor(data).cpu())
                new_labels.append(int(label))

        self.memory_set["data"].extend(new_data)
        self.memory_set["labels"].extend(new_labels)

        classes_in_memory = len(set(self.memory_set["labels"]))
        print(
            f"Memory updated: {len(self.memory_set['data'])} samples "
            f"across {classes_in_memory} classes "
            f"({samples_per_class}/class this task)"
        )
