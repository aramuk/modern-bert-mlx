import operator
from typing import Iterable, Union

import mlx as mx
import mlx.nn as nn
from rich import print

__all__ = ["ModuleList"]


class ModuleList(nn.Module):
    """Integer-indexable collection of modules, intended to mime torch.nn.ModuleList.

    Heavily borrows from: https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#ModuleList.
    """

    def __init__(self, modules: Iterable[nn.Module]):
        super().__init__()
        for idx, mod in enumerate(modules):
            setattr(self, str(idx), mod)

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: Union[int, slice]) -> Union[nn.Module, "ModuleList"]:
        if isinstance(idx, slice):
            return self.__class__(list(self.values())[idx])
        else:
            return self.get(self._get_abs_string_index(idx))


if __name__ == "__main__":
    ml = ModuleList([nn.Linear(4, 8) for _ in range(3)])
    assert ml[0] == ml.children()["0"]
    assert ml[-1] == ml.children()["2"]
    assert ml[0:2] == ModuleList([ml[0], ml[1]])
    assert ml[::-1] == ModuleList([ml[2], ml[1], ml[0]])
