"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

# from maze_env import Maze
from the_cliff_maze_env_up import Maze
from RL_brain import SarsaTable
import matplotlib.pyplot as plt
import numpy as np

total_reward_list = []

SAVE_RESULT_ENABLED = False
DISPLAY_ENV = True
TOTAL_EPISODES = 100

def update():
    for episode in range(TOTAL_EPISODES):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # total reward for this episode
        total_reward = 0
        time_steps = 0

        while True:
            # fresh env
            if(DISPLAY_ENV == True):
                env.render()

            time_steps += 1

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)
            total_reward += reward

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                print("Episode #{} finished after {} timesteps, total rewards = {}".format(episode, time_steps, total_reward))
                total_reward_list.append(total_reward)
                break

    # end of game + plot learning performance
    print('game over')
    env.destroy()
    plt.plot(np.arange(0,TOTAL_EPISODES), total_reward_list)
    plt.xlabel("episodes")
    plt.ylabel("total_rewards")
    #plt.xlim()
    #plt.ylim(-101,2)
    plt.title("Learning performance")

    if(SAVE_RESULT_ENABLED == True):
        plt.savefig("filename{}.png".format(TOTAL_EPISODES), format="png")
    else:
        plt.show()
    

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()