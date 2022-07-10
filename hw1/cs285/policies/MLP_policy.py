import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch.distributions import Categorical
from torch.distributions import Normal

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    '''
        - An abstract base class (ABC).
        - It inherits from: 
            - BasePolicy class in "policies/base_policy.py"
            - nn.Module from torch.nn
        - It overrides the methods:
            - get_action() from BasePolicy class
            - update() from BasePolicy class
            - save() from BasePolicy class
    '''

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        '''
            Intialize parameters for neural network
            for policy.

            params: 
                - they're all self-explanatory. Read them.
        '''
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # If obs/action space is discrete
        # no nead mean/std
        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)

        # If obs/action space is continuous
        # need to parametrize mean/std
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    # we're saving states? - Fade
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        # TODO (DONE!)
        # return the action that the policy prescribes
        obs = ptu.from_numpy(obs)
        with torch.no_grad():
            act = self(obs).sample()
        return ptu.to_numpy(act)
        # the LoadedGaussianPolicy implementation of get_action uses
        # self(observation). I still don't understand why you wrap it with self.

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # TODO (DONE!)
    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if self.discrete:
            return Categorical(logits=self.logits_na(observation))
        else:
            return Normal(self.mean_net(observation), torch.exp(self.logstd))


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO (DONE!) 
        # update the policy and return the loss
        actions = ptu.from_numpy(actions)
        observations = ptu.from_numpy(observations)
        act = self(observations).rsample()
        loss = self.loss(act, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
