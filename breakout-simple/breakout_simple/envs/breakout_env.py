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


    def create_obs_space(self):
        '''
        for pretty printing-keep size less than 10.
        '''
        G, W, R, P = ['G', 'X', '\u00B7', '\u15E7']
        as_ = np.empty(shape=[self.size, self.size], dtype='<U1')
        as_[:] = ' '
        N_ghosts, N_walls, N_reward = int(1*self.size), int(8*self.size), int(9*self.size)
        sum_ = N_ghosts + N_reward + N_walls
        pos = lambda x: (np.random.randint(x), np.random.randint(x))
        elem, count = [], 1
        while len(elem)<=sum_+1:
            sample = pos(self.size)
            if sample not in elem:
                s = pos(self.size)
                elem.append(s)
                if count<=N_ghosts: as_[s] = G #ghosts
                if N_ghosts < count <= N_ghosts + N_walls: as_[s] = W#walls
                if count > N_ghosts + N_walls: as_[s] = R  # reward
                if count == sum_+2: as_[s] = P #pacman
                count+=1
        y=np.empty((self.size+2,self.size+2), dtype='<U1')
        as_[as_ == ''] = ' '
        y[:]='X'
        y[1:self.size+1,1:self.size+1]=as_
        return y

    def pretty_print(self):

        y=self.observation_space
        pp = lambda x: print(re.sub('[\'[\]]', '', np.array_str(x)))
        for i in y: pp(i)



"""i make a new file  like cartpole.py and i used only the name of 3 function:init step and reset.
 Into "init" i have inizialize an array of array (matrix 10x10) and i choose where put pacman,ghost,wall, etc.... (like fully observable environment).
Into" step" i wrote the rules of the game, for example how my pacman must move and what happen if my pacman meet ghosts or walls.
Into "reset" i have specify what happen when my agent will dead.
this is what my friend said"""
