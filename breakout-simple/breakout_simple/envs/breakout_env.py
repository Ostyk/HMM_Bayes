import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import re

class BreakoutEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size):
        '''init space'''
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.create_obs_space()
        self._seed()

    def _seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        #actions is UP, DOWN, LEFT, RIGHT  = [0,1,2,3]
        '''rules of the game'''
        G, W, R, P = ['G', 'X', '\u00B7', '\u15E7']
        new_pos = self._take_action(action)
        #if self.observation_space(new_pos) == R:
        #    print("reward")
        #self.status = self.step()
        reward = self._get_reward(new_pos)
        ob = self.observation_space
        return ob, reward

    def _reset(self):
        '''what happens when we lose'''
        print("YOU DIED!")
        pass

    def _render(self, mode='human', close=False):
        self.pretty_print()

    def _take_action(self, action):
        G, W, R, P = ['G', 'X', '\u00B7', '\u15E7']
        #ghost, wall, reward,
        d = {0:"UP", 1:"DOWN", 2:"LEFT", 3:"RIGHT",}
        print("MOVE: {}\n".format(d[action]))
        #print(self.observation_space.shape)
        #x, y = np.array(np.where(self.observation_space=='\u15E7')).reshape(1,-1)[0][0]
        x,y = np.where(self.observation_space=='\u15E7')
        #print(x,y)
        new_pos = None
        if action == 0: #up
            value_check = self.observation_space[x-1, y]
            if value_check!=W:
                new_pos = [x-1, y]
        if action == 1: #down
            value_check = self.observation_space[x+1, y]
            if value_check!=W:
                new_pos = [x+1, y]
        if action == 2: #left
            value_check = self.observation_space[x, y-1]
            if value_check!=W:
                new_pos = [x, y-1]
        if action == 3: #right
            value_check = self.observation_space[x, y+1]
            if value_check!=W:
                new_pos = [x, y+1]
        return new_pos

    def _get_reward(self,new_pos):
        G, W, R, P = ['G', 'X', '\u00B7', '\u15E7']
        if new_pos is not None:
             in_cell = self.observation_space[tuple(new_pos)]
             if in_cell == G:
                 self._reset()
                 return 9 #game over
             elif in_cell==R: #next block is a reward
                 x,y = np.where(self.observation_space==P)
                 self.observation_space[x,y] = ' '
                 self.observation_space[tuple(new_pos)] = P
                 return 1
             else: #next block is empty
                 x,y = np.where(self.observation_space==P)
                 self.observation_space[x,y] = ' '
                 self.observation_space[tuple(new_pos)] = P
                 return 0
        else:
            return 0


    def create_obs_space(self, random_ = False):
        '''
        for pretty printing-keep size less than 10.
        '''
        G, W, R, P = ['G', 'X', '\u00B7', '\u15E7']
        as_ = np.empty(shape=[self.size, self.size], dtype='<U1')
        as_[:] = ' '
        for i in range(1,5,2):
            as_[1:5][:,i] = W
            as_[7:9][:,i] = W
        for i in range(6,10,2):
            as_[1:5][:,i] = W
            as_[7:9][:,i] = W

        N_ghosts, N_reward = int(1*self.size/2), int(5*self.size)
        sum_ = N_ghosts + N_reward
        pos = lambda x: (np.random.randint(x), np.random.randint(x))
        elem, ind = [], 1
        while ind < (sum_ + 1):
            sample = pos(self.size)
            if sample not in elem and as_[sample]==' ':
                elem.append(sample)
                if ind<=N_ghosts:
                    as_[sample] = G #ghosts
                elif N_ghosts < ind < sum_:
                    as_[sample] = R  # reward
                elif ind == sum_:
                    as_[sample] = P #pacman
                ind+=1

        y=np.empty((self.size+2,self.size+2), dtype='<U1')
        as_[as_ == ''] = ' '
        y[:]='X'
        y[1:self.size+1,1:self.size+1]=as_
        return y

    def pretty_print(self):

        y=self.observation_space
        pp = lambda x: print(re.sub('[\'[\]]', '', np.array_str(x)))
        for i in y: pp(i)
