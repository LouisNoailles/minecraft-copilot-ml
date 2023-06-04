import os
from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
from litemapy import Schematic
from nbtschematic import SchematicFile
import torch


def get_list_of_files(path: str) -> list[str]:
    list_files = os.listdir(path)
    concat_files = [os.path.join(path, f) for f in list_files]
    abs_path = [os.path.abspath(f) for f in concat_files]
    return abs_path


def random_block_destroyer(X: np.ndarray) -> np.ndarray:
    return X.copy()


class MinecraftSchematicsDataset(Dataset):
    def __init__(self, schematic_files: list[str]):
        self.schematic_files = schematic_files

    def __len__(self) -> int:
        return len(self.schematic_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            print(f"Loading {self.schematic_files[idx]} schematic as .litematic")
            schem = Schematic.load(self.schematic_files[idx])
            X: np.ndarray = list(schem.regions.values())[0]._Region__blocks
            X, y = np.copy(X), np.copy(X)
            # y = random_block_destroyer(X)
            return torch.from_numpy(X), torch.from_numpy(y)
        except Exception:
            print("Failing to load schematic as .litematic")
        try:
            print(f"Loading {self.schematic_files[idx]} schematic as .schematic")
            sf = SchematicFile.load(self.schematic_files[idx])
            X = np.asarray(sf.blocks)
            X, y = X.copy(), X.copy()
            # y = random_block_destroyer(X)
            return torch.from_numpy(X), torch.from_numpy(y)
        except Exception:
            print("Failing to load schematic as .schematic")
        raise Exception("Failed to load schematic")
