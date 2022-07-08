import abc
import numpy as np


class BasePolicy(object, metaclass=abc.ABCMeta):
    '''
        An abstract base class (ABC). Some methods will be
        implemented by a class 'MLPPolicySL' and some by
        'MLPPolicy' in file "policies/MLP_policy.py"
    '''
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError

# READ MORE ABOUT ABC here: https://www.geeksforgeeks.org/abstract-base-class-abc-in-python/
