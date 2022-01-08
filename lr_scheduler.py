import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class TransformerLR(_LRScheduler):
    """
    Attention Is All You Need §5.3 - Optimizer
    """

    def __init__(
            self,
            optimizer: Optimizer,
            d_model: int,
            warmup_steps: int = 4_000,
            last_epoch: int = -1,
            verbose: bool = False
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        super(self.__class__, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [
                group['lr'] for group in self.optimizer.param_groups
            ]

        lr_scale = self._get_lr_scale()
        return [
            group['lr'] * lr_scale for group in self.optimizer.param_groups
        ]

    def _get_lr_scale(self) -> float:
        return (self.d_model ** -0.5) * min(
            self.last_epoch ** -0.5,
            self.last_epoch * (self.warmup_steps ** -1.5)
        )
