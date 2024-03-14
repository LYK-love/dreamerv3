import numpy as np


class RandomAgent:

  def __init__(self, act_space):
    self.act_space = act_space

  def policy(self, obs, state=None, mode='train'):
    '''
    This won't change the state of the env. Because it's a policy. Policy will leverage the state, but will not change it.
    '''
    # Determines the batch size by looking at the length of the arrays contained in the obs dictionary. This is needed because the policy has to generate actions for each instance in the batch of observations. 
    # The next(iter(obs.values())) expression gets the first item from the values of obs, assuming all values (observations) have the same batch size.
    batch_size = len(next(iter(obs.values())))
    
    # The method iterates over self.act_space.items(), excluding any action type named 'reset', since 'reset' actions are likely managed separately and not part of the random action generation process.
    # For each action type in the action space, it **samples a random action batch_size times** using the sample() method provided by the action space. 
    # Recalling that the sample() method provided by the action space is a rendom sampling.
    # This results in a list of random actions for each action type.
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    return act, state
