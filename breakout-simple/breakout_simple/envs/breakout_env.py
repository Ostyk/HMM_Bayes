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
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        return ob, reward

    def _reset(self):
        '''what happens when we lose'''
        pass

    def _render(self, mode='human', close=False):
        self.pretty_print()


    def _take_action(self, action):
        G, W, R, P = ['G', 'X', '\u00B7', '\u15E7']
        #ghost, wall, reward,
        d = {0:"UP", 1:"DOWN", 2:"LEFT", 3:"RIGHT",}
        print("CURRENT ACTION: {}\n".format(d[action]))
        print(self.observation_space.shape)
        #x, y = np.array(np.where(self.observation_space=='\u15E7')).reshape(1,-1)[0][0]
        x,y = np.where(self.observation_space=='\u15E7')
        print(x,y)
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
        print(new_pos)
        return self.observation_space



        pass

    def _get_reward(self):
        """ Reward is given for XY. """
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return


    def create_obs_space(self):
        '''
        for pretty printing-keep size less than 10.
        '''
        G, W, R, P = ['G', 'X', '\u00B7', '\u15E7']
        as_ = np.empty(shape=[self.size, self.size], dtype='<U1')
        as_[:] = ' '
        N_ghosts, N_walls, N_reward = int(1*self.size), int(4*self.size), int(7*self.size)
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
