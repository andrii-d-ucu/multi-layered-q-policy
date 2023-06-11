from scipy import spatial
from scipy.spatial import distance
import numpy as np

class MultiLayeredQPolicy():
    def __init__(self):
        self.state_vectors = list()
        self.action_vectors = list()

        self.status = False
        self.previous_index = None
        self.state_model = None
        self.action_model = None
        self.memories = {}
        self.action_rewards = {}

    ## Method to fit state to action pairs data for k-d trees
    def fit(self, state_action_pairs):
        for pair in state_action_pairs:
            state, action = pair

            self.state_vectors.append(state)
            self.action_vectors.append(action)
        
        self.state_model = spatial.KDTree(self.state_vectors)
        self.action_model = spatial.KDTree(self.action_vectors)
    
    ## Getting action for state by using First and Second Layers
    def predict_action(self, state):
        if self.state_model == None:
            return None, 0, None, None
        
        state_vector, state_index = self.state_model.query(state)

        preferable_action_index = self.recall(state_index)
        state_distance  = 1 - distance.cosine(state_vector, state)

        ## First Layer Training
        if preferable_action_index == None:
            human_nearest_action = self.action_vectors[state_index]
            _, action_indexes = self.action_model.query(human_nearest_action, k=3)
            action_max_reward = None
            selected_action_index = None
            for action_index in action_indexes:

                if action_index not in self.action_rewards:
                    self.action_rewards[action_index] = [0]

                current_action_average_reward = np.mean(self.action_rewards[action_index])

                if action_max_reward == None:
                    action_max_reward = current_action_average_reward
                    selected_action_index = action_index
                elif current_action_average_reward > action_max_reward:
                    action_max_reward = current_action_average_reward
                    selected_action_index = action_index

            selected_action = self.action_vectors[selected_action_index]
            return selected_action, state_distance, state_index, selected_action_index
        else:
        ## Second Layer Usage
            selected_action = self.action_vectors[preferable_action_index]
            return selected_action, state_distance, state_index, preferable_action_index
    

    ## Memorazing sequence related data
    def remember(self, state_index, action_index, sequence_length):

        if state_index not in self.memories:
            self.memories[state_index] = {}

        if action_index not in self.memories[state_index]:
            self.memories[state_index][action_index] = []

        memory_data = self.memories[state_index][action_index]
        if len(memory_data) < 5:
            memory_data.append(sequence_length)
        else:
            memory_data.pop(0)
            memory_data.append(sequence_length)
        
        self.memories[state_index][action_index] = memory_data

    ## Selecting the best action from Second Layer standpoint
    def recall(self, state_index):
        
        if state_index not in self.memories:
            return None

        preferable_action_index = None
        preferable_action_reward = 0
        for action_index in self.memories[state_index]:
            memory_data = self.memories[state_index][action_index]
            reward = sum(memory_data) / len(memory_data)
            if preferable_action_reward < reward:
                preferable_action_reward = reward
                preferable_action_index = action_index

        return preferable_action_index
    

    ## Updating reward for First Layer
    def update_action_reward(self, action_index, reward):
        if action_index not in self.action_rewards:
            self.action_rewards[action_index] = []
        if len(self.action_rewards[action_index]) < 5:
            self.action_rewards[action_index].append(reward)
        else:
            self.action_rewards[action_index].pop(0)
            self.action_rewards[action_index].append(reward)
    
    ## Updating reward for Second Layer
    def update_memory(self, state_action_pairs, sequence_length):

        for state_action_pair in state_action_pairs:
            state_index, action_index = state_action_pair
            self.remember(state_index, action_index, sequence_length)