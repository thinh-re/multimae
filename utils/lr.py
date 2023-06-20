from abc import ABCMeta, abstractmethod
from typing import Dict


class BaseLR:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter: int):
        pass

    def state_dict(self) -> Dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)


class PolyLR(BaseLR):
    def __init__(self, start_lr: float, lr_power: float, total_iters: int):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter: int) -> float:
        return self.start_lr * (
            (1 - float(cur_iter) / self.total_iters) ** self.lr_power
        )


class LinearLR(BaseLR):
    def __init__(self, start_lr: float, end_lr: float, total_iters: int):
        """@Deprecated"""
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_iters = float(total_iters)

        self.b = self.start_lr
        self.a = (self.end_lr - self.start_lr) / self.total_iters

    def get_lr(self, cur_iter: int) -> float:
        return self.a * cur_iter + self.b


class LinearLRRestart(BaseLR):
    def __init__(
        self,
        start_lr: float,
        end_lr: float,
        num_epochs_every_restart: int,
    ):
        """Note: Remember to set epoch at the begining of each epoch"""
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs_every_restart = num_epochs_every_restart

    def set_epoch(self, epoch: int, total_iters_per_epoch: int) -> None:
        """
        if epoch is between 1->100, upperbound will be 100
        if epoch is between 101->200, upperbound will be 200
        """
        upperbound = (
            ((epoch - 1) // self.num_epochs_every_restart) + 1
        ) * self.num_epochs_every_restart
        total_iters = upperbound * total_iters_per_epoch

        self.b = self.start_lr
        self.a = (self.end_lr - self.start_lr) / total_iters

    def get_lr(self, cur_iter: int) -> float:
        """Note: the beginning cur_iter is 0"""
        return self.a * cur_iter + self.b

    def __item__(self, cur_iter: int) -> float:
        return self.get_lr(cur_iter)
