import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct


from agent import Agent

import gym
import numpy as np


import time

game=gym.make('CartPole-v0')

input_size=game.observation_space.shape[0]
output_size=game.action_space.n
gamma=.9
epsilon=.1

agent=Agent(input_size,500,output_size,50,epsilon,gamma)

action_skips=1


past_rewards=[]
past_size=20


while True:
    state=game.reset()
    total_rewards=0
    
    for i in range(10000):

        game.render()
        action,dex=agent.get_action(state)

        new_state,reward,done,_ = game.step(dex)
        
        total_rewards+=reward
        

        # time.sleep(.05)


        if done or i == 500:
            total_rewards-=5

            agent.n_games+=1


            agent.remember(state,action,total_rewards,new_state,done)
            agent.train_short_memory(state,action,total_rewards,new_state,done)

            past_rewards.append(total_rewards)
            
            if len(past_rewards)>past_size:
                past_rewards.pop(0)

            print('*'*50)
            print(agent.get_ratios(),'- game#: ',agent.n_games, '- total reward: ',total_rewards)
            print('average ',round(sum(past_rewards)/len(past_rewards),4),'-','reward: ',reward)
            print('chosen: ',dex,'- actions: ',[i.item() for i in action])
            print('*'*50)

            agent.reset_ratios()
            break

        else:

            if i%action_skips==0:
                agent.remember(state,action,total_rewards,new_state,done)
                agent.train_short_memory(state,action,total_rewards,new_state,done)
                
                
                
                

            state=new_state

    agent.train_long_memory()

    if agent.n_games>=10000:
        break




game.close() 














