import gym
import os
if not os.path.split(os.getcwd())[1]=='breakout-simple':
    os.chdir('breakout-simple')
import breakout_simple

env = gym.make('breakout-simple-v0')


if __name__ == '__main__':
    import sys
    val = int(sys.argv[1])
    r=0
    print("INITIAL")
    env.render()
    print("_"*40)
    print("_"*40)
    while r<val:
        o, rew = env._step(env.action_space.sample())
        if rew is not None:
            r+=1
        env.render()
        print("total reward:", r)
        print("_"*40)
