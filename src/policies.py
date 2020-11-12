import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Optional


class BasePolicy:
    "Base class of Policy"

    def __call__(self):
        raise NotImplementedError


class DiscreteCategoricalDistributionPolicy(BasePolicy):
    """
    Sample actions from a given distribution

    Parameters:
        net: torch module to output the probabilities
        device: device where the calculations are performed ['cpu', 'cuda']
        apply_softmax: apply softmax function to the net output, required if the net
            outputs logits instead of distributions.
        net_index: if the net has more than one output, specify the index corresponding
            to the probabilities output.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        device: Union[str, torch.device],
        apply_softmax: bool = False,
        net_index: Optional[int] = None,
        add_batch_dim: bool = False,
    ):
        self.net = net
        self.device = device
        self.apply_softmax = apply_softmax
        self.net_index = net_index
        self.add_batch_dim = add_batch_dim

    @torch.no_grad()
    def __call__(self, states: np.ndarray):
        if self.add_batch_dim:
            states = states[np.newaxis, :]
        states_v = torch.tensor(states, dtype=torch.float32).to(self.device)
        probs_t = self._get_probs(states_v)
        if self.apply_softmax:
            probs_t = F.softmax(probs_t, dim=1)
        probs_dist = torch.distributions.Categorical(probs_t)

        if self.add_batch_dim:
            return probs_dist.sample().cpu().numpy()[0]
        return probs_dist.sample().cpu().numpy()

    def _get_probs(self, states_v: torch.Tensor):
        probs_v = self.net(states_v)
        if self.net_index is not None:
            return probs_v[self.net_index]
        return probs_v


class DiscreteRandomPolicy(DiscreteCategoricalDistributionPolicy):
    """
    Sample actions randomly

    Parameters:
        net: torch module to output the probabilities
        device: device where the calculations are performed ['cpu', 'cuda']
        net_index: if the net has more than one output, specify the index corresponding
            to the probabilities output.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        device: Union[str, torch.device],
        net_index: Optional[int] = None,
        add_batch_dim: bool = False,
    ):
        super().__init__(
            net=net,
            device=device,
            apply_softmax=True,
            net_index=net_index,
            add_batch_dim=add_batch_dim,
        )

    def _get_probs(self, states_v: torch.Tensor):
        states_v = torch.rand_like(states_v).to(self.device)
        probs_v = self.net(states_v)
        if self.net_index is not None:
            return probs_v[self.net_index]
        return probs_v


class DiscreteGreedyPolicy(DiscreteCategoricalDistributionPolicy):
    """
    Sample actions by taking the argmax operation

    Parameters:
        net: torch module to output the probabilities
        device: device where the calculations are performed ['cpu', 'cuda']
        apply_softmax: apply softmax function to the net output, required if the net
        net_index: if the net has more than one output, specify the index corresponding
            to the probabilities output.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        device: Union[str, torch.device],
        net_index: Optional[int] = None,
        add_batch_dim: bool = False,
    ):
        super().__init__(
            net=net,
            device=device,
            apply_softmax=False,
            net_index=net_index,
            add_batch_dim=add_batch_dim,
        )

    @torch.no_grad()
    def __call__(self, states: np.ndarray):
        if self.add_batch_dim:
            states = states[np.newaxis, :]
        states_v = torch.tensor(states, dtype=torch.float32).to(self.device)
        probs_t = self._get_probs(states_v)
        argmax = torch.argmax(probs_t, dim=1)

        if self.add_batch_dim:
            return argmax.cpu().numpy()[0]
        return argmax.cpu().numpy()


class GaussianPolicy(BasePolicy):
    def __init__(self, net: torch.nn.Module, device: Union[str, torch.device]):
        self.net = net
        self.device = device

    @torch.no_grad()
    def __call__(self, states: np.ndarray):
        states_v = torch.tensor(states, dtype=torch.float32).to(self.device)

        mean_t, std_t, _ = self.net(states_v)
        actions = torch.normal(mean_t, std_t).squeeze()

        return actions.cpu().numpy()


if __name__ == "__main__":
    pass
