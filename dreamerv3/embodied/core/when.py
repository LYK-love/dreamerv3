import time


class Every:

  def __init__(self, every, initial=True):
    self._every = every
    self._initial = initial
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    if self._prev is None:
      self._prev = (step // self._every) * self._every
      return self._initial
    if step >= self._prev + self._every:
      self._prev += self._every
      return True
    return False


class Ratio:

  def __init__(self, ratio):
    '''
    The constructor takes a single argument ratio, which represents the desired frequency of the action relative to the number of steps. 
    The ratio is stored in _ratio, and _prev is initialized to None. 
    _prev is used to keep track of the step number the last time the action was performed (or decided to be performed).
    '''
    
    assert ratio >= 0, ratio
    self._ratio = ratio
    self._prev = None

  def __call__(self, step):
    '''
    This method is called with the current step number as its argument. It calculates how many times the action should be repeated based on the elapsed steps since the last action and the specified ratio.

    If _ratio is 0, the function immediately returns 0, indicating that the action should never be performed.
    If _prev is None (indicating this is the first time the method is called), it sets _prev to the current step and returns 1, signifying that the action should be performed now.
    Otherwise, it calculates the number of repeats as (step - self._prev) * self._ratio, updates _prev to reflect the new position, and returns the integer number of repeats. This effectively spaces out the actions according to the specified ratio relative to the number of steps.
    '''
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    step = int(step)
    if not self._until:
      return True
    return step < self._until


class Clock:

  def __init__(self, every):
    self._every = every
    self._prev = None

  def __call__(self, step=None):
    '''
    Use `save_every` to judge whether to save the checkpoint.
    '''
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    now = time.time()
    if self._prev is None:
      self._prev = now
      return True
    if now >= self._prev + self._every:
      # self._prev += self._every
      self._prev = now
      return True
    return False
