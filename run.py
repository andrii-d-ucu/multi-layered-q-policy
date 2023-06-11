import statistics
from osim_environment import OsimEnvironment
import copy
import numpy as np
import pandas as pd
import traceback
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
from core import MultiLayeredQPolicy

TRAINING_EPOCHS = 3
EXPLOITATION_EPOCHS = 10
ENV_MEMORY_SIZE = 1
INTEGRATOR_ACCURACY = 1e-3
EPISODES_PER_EPOCH = 4

mode = '2D'
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))

if mode is '2D':
    params = np.loadtxt('./osim/control/params_2D.txt')
elif mode is '3D':
    params = np.loadtxt('./osim/control/params_3D_init.txt')

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)

train_steps = 25

policy = MultiLayeredQPolicy()

env = OsimEnvironment(visualize=True, memory_size=ENV_MEMORY_SIZE, integrator_accuracy=INTEGRATOR_ACCURACY)
env.change_model()


print("training model...")
for epoch in range(TRAINING_EPOCHS):
    try:
        rewards = []
        for num_episode in range(EPISODES_PER_EPOCH):
            t = 0
            done = False
            state_distance = 0
            action_counter = 0
            state_action_vector_pairs = []
            r = 0

            obs_dict, observation, concated_vector = env.reset()
            env.spec.timestep_limit = timstep_limit

            while not done:
                previous_concated_state_vector = copy.deepcopy(concated_vector)

                #creating human action
                t += sim_dt
                locoCtrl.set_control_params(params)
                human_action = locoCtrl.update(obs_dict)
                
                #creating agent action
                agent_action, state_distance, state_index, action_index = policy.predict_action(previous_concated_state_vector)

                action_counter += 1
                
                #applying action
                finalized_action = None
                if state_distance > 0.5 and agent_action != None:
                    finalized_action = agent_action
                    obs_dict, observation, concated_vector, reward, done, info = env.step(agent_action)
                    policy.update_action_reward(action_index, reward)
                else:
                    finalized_action = human_action
                    obs_dict, observation, concated_vector, reward, done, info = env.step(human_action)
                

                r += reward

                state_action_vector_pairs.append((previous_concated_state_vector, finalized_action))
                
                if action_counter >= 100:
                    done = True

            print('Finished episode ', num_episode)
            rewards.append(r)
            policy.fit(state_action_vector_pairs)

        print('Epoch: {0}, Avg rew: {1}'.format(epoch, statistics.mean(rewards)))
    except:
        traceback.print_exc()

print("exploiting model")
for epoch in range(EXPLOITATION_EPOCHS):
    try:
        rewards = []
        for num_episode in range(EPISODES_PER_EPOCH):

            t = 0
            done = False
            state_distance = 0
            action_counter = 0
            state_action_pairs = []
            r = 0

            obs_dict, observation, concated_vector = env.reset()
            env.spec.timestep_limit = timstep_limit

            while not done:
                action_agent, state_distance, state_index, action_index = policy.predict_action(concated_vector)
                state_action_pairs.append((state_index, action_index))
                obs_dict, observation, concated_vector, reward, done, info = env.step(action_agent)
                r += reward
                action_counter += 1

                if action_counter >= 100:
                    done = True

            policy.update_memory(state_action_pairs, action_counter)
            print('Finished episode ', num_episode)
            rewards.append(r)

        print('Epoch: {0}, Avg rew: {1}'.format(epoch, statistics.mean(rewards)))
    except:
        traceback.print_exc()