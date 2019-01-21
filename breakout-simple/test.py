import gym
import os
if not os.path.split(os.getcwd())[1]=='breakout-simple':
    os.chdir('breakout-simple')
import breakout_simple
import time



if __name__ == '__main__':
    import sys
    val = int(sys.argv[1])
    rew=0
    while rew != 9:
        env = gym.make('breakout-simple-v0')
        print("INITIAL")
        print("_"*40)
        r=0
        while r<val:
            env.render()
            o, rew = env._step(env.action_space.sample())
            if rew == 9:
                os.system('clear')
                print("You died, let's try again")
                r=0
                time.sleep(1)

            if rew == 1: r+=1
            print("total reward:", r)
            print("_"*40)
            time.sleep(0.5)
        if r==5:
            os.system('clear')
            print("YOU WIN")
            break
