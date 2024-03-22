import re

import embodied
import numpy as np


def train(agent, env, replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps) # In practice, train_ratio is overrided to be 32. # 32 / (16 * 64) = 1/32
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  # Decorates these methods.
  # The original methods will be replaced by the decorators which can do timing.
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum()) # summation of rewards
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum()) # # summation of absolute values of rewards
    logger.add({
        'length': length,
        'score': score,
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key] # Record gt video??
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      # add log
      # log_keys_sum and log_keys_max won't be matched since their regexes are `^$`.
      # So they won't be logged.
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  #########################################
  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add) # Before each update step, collect the step data to the replay buffer.
  # Use a random agent to initialize the replay buffer.
  print('Prefill train dataset.')
  random_agent = embodied.RandomAgent(env.act_space)
  while len(replay) < max(args.batch_steps, args.train_fill): # 
    print(f"length of replay buffer: {len(replay)}")
    driver(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  d = replay.dataset # Generater of the replay buffer
  dataset = agent.dataset(replay.dataset) # Use the generator to get the dataset
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    '''
    Although we have `tran` as param. We don't use it. In training we only use the samples from buffer.
    '''
    repeats = should_train(step) # Either 0 (not training) or 1 (traiing).
    # print(f"repeats: {repeats}")
    
    # If you have args.train_ratio = 3 and args.batch_steps = 16, then args.train_ratio / args.batch_steps gives 0.1875. 
    # This means, on average, you want the training to happen once every approximately 5.33 (1 / 0.1875) steps.
    for _ in range(repeats):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
      # OK. So in each "update step", there will be one traning (dyn + behavior)
      # The input data `batch[0]` is sampled from the replay buffer.
      # THe initial state is None.
      outs, state[0], mets = agent.train(batch[0], state[0])  # The training
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      agg = metrics.result()
      report = agent.report(batch[0]) 
      report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      logger.add(report, prefix='report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step) # Each step call `train_step` one time

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint: # By default `from_checkpoint`=False.
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we justd saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  
  # The outermost loop
  while step < args.steps:
    driver(policy, steps=100) # C=100 by default.
    if should_save(step):
      checkpoint.save()
  logger.write()
