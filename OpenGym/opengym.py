# opengym.py
"""Volume 2: Open Gym
Caelan Osman
Math 323 Sec. 2
Feb 18, 2021
"""

import gym
import numpy as np
from IPython.display import clear_output
import random

def find_qvalues(env,alpha=.1,gamma=.6,epsilon=.1):
    """
    Use the Q-learning algorithm to find qvalues.

    Parameters:
        env (str): environment name
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): maximum value

    Returns:
        q_table (ndarray nxm)
    """
    # Make environment
    env = gym.make(env)
    # Make Q-table
    q_table = np.zeros((env.observation_space.n,env.action_space.n))

    # Train
    for i in range(1,100001):
        # Reset state
        state = env.reset()

        epochs, penalties, reward, = 0,0,0
        done = False

        while not done:
            # Accept based on alpha
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, done, info = env.step(action)

            # Calculate new qvalue
            old_value = q_table[state,action]
            next_max = np.max(q_table[next_state])

            new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            # Check if penalty is made
            if reward == -10:
                penalties += 1

            # Get next observation
            state = next_state
            epochs += 1

        # Print episode number
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.")
    return q_table

# Problem 1
def random_blackjack(n):
    """
    Play a random game of Blackjack. Determine the
    percentage the player wins out of n times.

    Parameters:
        n (int): number of iterations

    Returns:
        percent (float): percentage that the player
                         wins
    """
    #create enviroment
    env = gym.make("Blackjack-v0")
    #how many times we ahve won
    won = 0
    #play blacjack n times
    for _ in range(n):
        #reset the enviroment
        env.reset()
        done = False
        #while not done draw card
        while not done:
            observation, reward, done, info = env.step(env.action_space.sample())

        #if the reward is 1 that means we won
        if reward == 1:
            won += 1

    #close the enviroment
    env.close()
    #return percentage of times we've won
    return won / n


# problem 2
def blackjack(n=11):
    """
    Play blackjack with naive algorithm.

    Parameters:
        n (int): maximum accepted player hand

    Return:
        percent (float): percentage of 10000 iterations
                         that the player wins
    """

    #naive algorithm to play blackjack 10k times
    env = gym.make('Blackjack-v0')
    def naive(n):
        observation = env.reset()
        hand = observation[0]
        done = False
        while not done:
            if hand <= n:
                observation, reward, done, _ = env.step(1)
                hand = observation[0]
            else:
                observation, reward, done, _ = env.step(0)
        #return 1 if we won zero if we didn't
        if reward == 1:
            return 1
        else:
            return 0

    #initialize count of won games
    won_count = 0
    #run 10k times
    for _ in range(10000):
        won_count += naive(n)
    env.close()

    #return percentage
    return won_count / 10000

# Problem 3
def cartpole():
    """
    Solve CartPole-v0 by checking the velocity
    of the tip of the pole

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    #make enviremont and get initial state
    env = gym.make('CartPole-v0')
    positions_velocities = env.reset()
    #initialize done to False
    done = False
    #indexing variables
    pole, right, left, num_steps = 3, 1, 0, 0
    #continue until the pole has fallen over
    while not done:
        #if pole velocity is nonpositive
        if positions_velocities[pole] <= 0:
            positions_velocities, reward, done, _ = env.step(left)
        #if pole velocity is positive
        else:
            positions_velocities, reward, done, _ = env.step(right)

        num_steps += reward

    return num_steps

# Problem 4
def car():
    """
    Solve MountainCar-v0 by checking the position
    of the car.

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    #make enviroment
    env = gym.make('MountainCar-v0')
    #reset and get current position
    position_velocity = env.reset()
    #initialize done to False and count to zero
    done, count = False, 0
    while not done:
        #always start by stepping to the right and continuing
        #to do so if velocity is positive
        if position_velocity[-1] >= 0:
            position_velocity, reward, done, _ = env.step(2)
        #step to the left once velocity starts getting negative
        else:
            position_velocity, reward, done, _ = env.step(0)
        #increment count
        count += 1

    return count

# Problem 5
def taxi(q_table):
    """
    Compare naive and q-learning algorithms.

    Parameters:
        q_table (ndarray nxm): table of qvalues

    Returns:
        naive (float): mean reward of naive algorithm
                       of 10000 runs
        q_reward (float): mean reward of Q-learning algorithm
                          of 10000 runs
    """

    #initialize enviroment
    env = gym.make('Taxi-v3')
    #naive random algorithm
    def naive_taxi():
        #reset enviroment
        env.reset()
        #intialize done to false and reward to zero
        reward, done = False, 0

        #take random actions
        while not done:
            _, value, done, _ = env.step(env.action_space.sample())
            reward += value

        #return reward
        return reward

    #algorithm for using qlearning
    def qlearn_taxi():
        #reset envireoment
        observation = env.reset()
        #intialize done and the reward 
        reward, done= False, 0
        #continue until the enviroment is finished
        while not done:
            #step using optimal index of the current row
            observation, value, done, _ = env.step(np.argmax(q_table[observation]))
            #increment reward
            reward += value
            #increment step

        return reward

    #total rewards for naive and qlearning methods
    naive = 0
    qlearn = 0
    #run 10k times
    for _ in range(10000):
        naive += naive_taxi()
        qlearn += qlearn_taxi()

    #close the enviroment
    env.close()
    #return averages
    return naive / 10000, qlearn / 10000


if __name__ == "__main__":


    #problem 1
    '''
    percentage = []
    for n in range(500, 10500, 500):
        percentage.append(random_blackjack(n))

    print(percentage)
    '''

    #prob 2
    '''
    percentage = np.array([blackjack(n) for n in range(21)])
    print(percentage)
    print(np.mean(percentage))
    '''

    #prob 3
    '''
    percentage = np.array([cartpole() for _ in range(21)])
    print(percentage)
    print(np.mean(percentage))
    '''

    #prob4
    #print(car())

    #prob 5
    '''
    qtable = find_qvalues('Taxi-v3')
    values = taxi(qtable)
    print(values[0])
    print(values[1])
    '''


