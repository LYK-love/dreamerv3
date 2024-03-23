import concurrent.futures
import time

from . import basics
from . import path


class Checkpoint:

  def __init__(self, filename=None, log=True, parallel=True):
    self._filename = filename and path.Path(filename)
    self._log = log
    self._values = {}
    self._parallel = parallel
    if self._parallel:
      self._worker = concurrent.futures.ThreadPoolExecutor(1)
      self._promise = None

  def __setattr__(self, name, value):
    '''
    This method is overridden to enforce that any attribute set on the Checkpoint object (not starting with _ and not being one of the excluded methods) must have both save() and load() methods. 
    This requirement ensures that only objects that can be serialized and deserialized are added to the checkpoint.
    '''
    if name in ('exists', 'save', 'load'):
      return super().__setattr__(name, value)
    if name.startswith('_'):
      return super().__setattr__(name, value)
    has_load = hasattr(value, 'load') and callable(value.load) # must have load method
    has_save = hasattr(value, 'save') and callable(value.save) # must have save method
    if not (has_load and has_save):
      message = f"Checkpoint entry '{name}' must implement save() and load()."
      raise ValueError(message)
    self._values[name] = value

  def __getattr__(self, name):
    '''
    Attempts to retrieve the attribute from the _values dictionary, raising an error if the attribute starts with _ or is not found.
    '''
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      return getattr(self._values, name)
    except AttributeError:
      raise ValueError(name)

  def exists(self, filename=None):
    '''
    Checks whether a checkpoint file exists at the specified or default location.
    '''
    assert self._filename or filename
    filename = path.Path(filename or self._filename)
    exists = self._filename.exists()
    self._log and exists and print('Found existing checkpoint.')
    self._log and not exists and print('Did not find any checkpoint.')
    return exists

  def save(self, filename=None, keys=None):
    '''
    Saves the current state of all objects registered with the checkpoint. 
    If parallel execution is enabled, the saving process is submitted as a task to the thread pool executor; otherwise, it's executed in the main thread. 
    The actual saving logic, including data serialization and file handling, is encapsulated in the _save method.
    '''
    assert self._filename or filename
    filename = path.Path(filename or self._filename)
    self._log and print(f'Writing checkpoint: {filename}')
    if self._parallel:
      self._promise and self._promise.result()
      self._promise = self._worker.submit(self._save, filename, keys)
    else:
      self._save(filename, keys)

  def _save(self, filename, keys):
    keys = tuple(self._values.keys() if keys is None else keys)
    assert all([not k.startswith('_') for k in keys]), keys
    data = {k: self._values[k].save() for k in keys}
    data['_timestamp'] = time.time()
    if filename.exists():
      old = filename.parent / (filename.name + '.old')
      filename.copy(old)
      filename.write(basics.pack(data), mode='wb')
      old.remove()
    else:
      filename.write(basics.pack(data), mode='wb')
    self._log and print(f'Wrote checkpoint: {filename}')

  def load(self, filename=None, keys=None):
    assert self._filename or filename
    filename = path.Path(filename or self._filename)
    self._log and print(f'Loading checkpoint: {filename}')
    data = basics.unpack(filename.read('rb'))
    keys = tuple(data.keys() if keys is None else keys)
    for key in keys:
      if key.startswith('_'):
        continue
      try:
        self._values[key].load(data[key])
      except Exception:
        print(f'Error loading {key} from checkpoint.')
        raise
    if self._log:
      age = time.time() - data['_timestamp']
      print(f'Loaded checkpoint from {age:.0f} seconds ago.')

  def load_or_save(self):
    if self.exists():
      self.load()
    else:
      self.save()
