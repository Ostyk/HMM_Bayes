import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
import numpy as np

register(
    id='breakout-simple-v0',
    entry_point='breakout_simple.envs:BreakoutEnv',
    nondeterministic = True,
    kwargs={'size' : 10},
)
