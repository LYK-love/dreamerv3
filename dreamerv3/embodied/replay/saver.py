import concurrent.futures
from collections import defaultdict, deque
from functools import partial as bind

import embodied

from . import chunk as chunklib


class Saver:

  def __init__(self, directory, chunks=1024):
    self.directory = embodied.Path(directory)
    self.directory.mkdirs()
    self.chunks = chunks
    self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
    self.workers = concurrent.futures.ThreadPoolExecutor(16)
    self.promises = deque()
    self.loading = False

  def add(self, step, worker):
    '''
    step: A single step or data point to be added.
    worker: Identifier for the source of the step, used to segregate data by its origin.
    If not loading data, it appends the step to the buffer associated with the given worker.
    Once a buffer reaches its capacity (self.chunks), it initiates a save operation for that buffer and resets the buffer for future data.
    '''
    if self.loading:
      return
    buffer = self.buffers[worker]
    buffer.append(step)
    if buffer.length >= self.chunks:
      self.buffers[worker] = buffer.successor = chunklib.Chunk(self.chunks)
      self.promises.append(self.workers.submit(buffer.save, self.directory))
      for promise in [x for x in self.promises if x.done()]:
        promise.result()
        self.promises.remove(promise)

  def save(self, wait=False):
    '''
    Initiates the saving of any buffer that has accumulated steps.
    If wait is True, it blocks until all scheduled save operations have completed.
    '''
    for buffer in self.buffers.values():
      if buffer.length:
        self.promises.append(self.workers.submit(buffer.save, self.directory))
    if wait:
      [x.result() for x in self.promises]
      self.promises.clear()

  def load(self, capacity, length):
    '''
    Scans the directory for saved chunks that fit within specified capacity and length criteria.
    Loads data chunks concurrently, re-establishing the ordering based on timestamps and ensuring continuity through streamids.
    Yields steps from loaded chunks one at a time for processing or insertion into the replay mechanism.
    '''
    # Scanning for Chunks:
    filenames = chunklib.Chunk.scan(self.directory, capacity, length - 1)
    if not filenames:
      return
    
    # Concurrent Loading:
    threads = min(len(filenames), 32) # Determines the number of threads to use for concurrent loading, capped at 32 to prevent overloading the system.
    with concurrent.futures.ThreadPoolExecutor(threads) as executor: # nitializes a thread pool executor with the determined number of threads.
      chunks = list(executor.map(chunklib.Chunk.load, filenames)) # Loads the chunk files concurrently using the executor. The load function is called for each filename in the filenames list, returning chunk objects which are then collected into a list called chunks.
    # Establishing Continuity through Stream IDs:
    # The next few lines establish continuity between chunks by assigning stream IDs based on the chunks' UUIDs and their successor relationships.
    streamids = {}
    
    # Chunks are sorted in reverse order based on their timestamps to ensure the latest chunks are processed first.
    for chunk in reversed(sorted(chunks, key=lambda x: x.time)):
      if chunk.successor not in streamids:
        streamids[chunk.uuid] = int(embodied.uuid())
      else:
        streamids[chunk.uuid] = streamids[chunk.successor]
    # Yielding Steps:
    self.loading = True # enters a loading state by setting self.loading = True.
    for i, chunk in enumerate(chunks): # It then iterates over each chunk and yields individual steps contained within the chunk. Each step is a dictionary of data items extracted from the chunk.
      stream = streamids[chunk.uuid] # Stream IDs are also yielded with each step to maintain the context of the data stream.
      for index in range(chunk.length):
        step = {k: v[index] for k, v in chunk.data.items()}
        yield step, stream
      # Free memory early to not require twice the replay capacity.
      chunks[i] = None # After a chunk is processed, it's explicitly set to None to free up memory early, considering that the entire replay capacity doesn't need to be held in memory at once.
      del chunk
    self.loading = False # After all chunks have been processed and their steps yielded, the method exits the loading state by setting self.loading = False.
