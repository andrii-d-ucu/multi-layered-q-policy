from osim.env import L2M2019Env
from collections import deque
import pickle

import numpy as np

INIT_POSE = np.array([
    1.699999999999999956e+00, # forward speed
    .5, # rightward speed
    9.023245653983965608e-01, # pelvis height
    2.012303881285582852e-01, # trunk lean
    0*np.pi/180, # [right] hip adduct
    -6.952390849304798115e-01, # hip flex
    -3.231075259785813891e-01, # knee extend
    1.709011708233401095e-01, # ankle flex
    0*np.pi/180, # [left] hip adduct
    -5.282323914341899296e-02, # hip flex
    -8.041966456860847323e-01, # knee extend
    -1.745329251994329478e-01]) # ankle flex

def leg_observation_to_vector(leg_obs):
    values = leg_obs['ground_reaction_forces']
    dict_keys = set(leg_obs.keys()) - {'ground_reaction_forces'}
    for key in dict_keys:
        values.extend(leg_obs[key].values())
    return np.array(values)


def observation2tensors(observation):
    tgt_field = observation['v_tgt_field']
    tgt_field = np.transpose(tgt_field, (1, 2, 0))

    pelvis = np.array([
                          observation['pelvis']['height'],
                          observation['pelvis']['pitch'],
                          observation['pelvis']['roll']
                      ] + observation['pelvis']['vel'])

    r_leg = leg_observation_to_vector(observation['r_leg'])
    l_leg = leg_observation_to_vector(observation['l_leg'])
    return tgt_field, np.concatenate((pelvis, r_leg, l_leg), axis=0)


class OsimEnvironment(L2M2019Env):
    def __init__(self, *args, memory_size=5, visualize=True, integrator_accuracy=3e-2, difficulty=1, seed=None,
                 report=None, **kwargs):
        super(OsimEnvironment, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy,
                                    difficulty=difficulty, seed=seed,
                                    report=report)
        self.tgt_field_queue = deque(maxlen=memory_size)
        self.body_vector_queue = deque(maxlen=memory_size)
        self.actions_queue = deque(maxlen=memory_size)
        self.reward_policy = None

    def change_model(self, model= '2D', difficulty=1, seed=None):
        super(OsimEnvironment, self).change_model(model=model, difficulty=difficulty, seed=seed)

    def set_reward_policy(self, reward_policy):
        self.reward_policy = reward_policy

    def reset(self, project=True, seed=None, init_pose=INIT_POSE, obs_as_dict=True):
        obs = super(OsimEnvironment, self).reset(project=True, seed=None, init_pose=init_pose, obs_as_dict=True)

        tgt_field, body_vector = observation2tensors(obs)

        for i in range(self.tgt_field_queue.maxlen):
            self.tgt_field_queue.append(tgt_field)
            self.body_vector_queue.append(body_vector)
            self.actions_queue.append(np.full((22), 0.05))
        
        tgt_field, body_vector = self.flatten_queues()

        tgt_field_flatten = tgt_field.flatten()
        concated_vector = np.concatenate([tgt_field_flatten, body_vector], axis=-1)

        return obs, (tgt_field, body_vector), concated_vector
    
        

    def step(self, action, project=True, obs_as_dict=True):

        obs, reward, done, info = super(OsimEnvironment, self).step(action, project=project, obs_as_dict=True)
        
        tgt_field, body_vector = observation2tensors(obs)

        self.tgt_field_queue.append(tgt_field)
        self.body_vector_queue.append(body_vector)
        self.actions_queue.append(action)

        tgt_field, body_vector = self.flatten_queues()    


        tgt_field_flatten = tgt_field.flatten()
        concated_vector = np.concatenate([tgt_field_flatten, body_vector], axis=-1)

        if self.reward_policy != None:
            modified_reward = reward * self.reward_policy.calculate(concated_vector, action)
            print(f"reward {modified_reward}")
            reward = modified_reward
            

        return obs, (tgt_field, body_vector), concated_vector, reward, done, info

    def flatten_queues(self):
        tgt_fields = [f for f in self.tgt_field_queue]
        body_vectors = [f for f in self.body_vector_queue]
        return np.concatenate(tgt_fields, axis=-1), np.concatenate(body_vectors, axis=0)
