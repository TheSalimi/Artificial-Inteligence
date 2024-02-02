import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
device = torch.device("cpu") 
import warnings
from torch.distributions import Categorical

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))

class memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class neural_network(nn.Module):
    def __init__(self,
                layers=[8,64,32,4],
                dropout=False,
                p_dropout=0.5,
                ):
        super(neural_network,self).__init__()

        self.network_layers = []
        n_layers = len(layers)
        for i,neurons_in_current_layer in enumerate(layers[:-1]):
            
            self.network_layers.append(nn.Linear(neurons_in_current_layer, 
                                                layers[i+1]) )
            if dropout:
                self.network_layers.append( nn.Dropout(p=p_dropout) )

            if i < n_layers - 2:
                self.network_layers.append( nn.ReLU() )
        
        self.network_layers = nn.Sequential(*self.network_layers)

    def forward(self,x):
        for layer in self.network_layers:
            x = layer(x)
        return x

class dqn():
    def __init__(self,parameters):
        self.n_state = parameters['n_state']
        self.n_actions = parameters['n_actions']

        parameters2 = {
            'neural_networks':
                {
                'policy_net':{
                    'layers':[self.n_state,128,32,self.n_actions],
                            }
                },
            'optimizers': #This key specifies the optimizer used to update the parameters of the neural network during training.
                {
                'policy_net':{
                    'optimizer':'RMSprop',
                     'optimizer_args':{'lr':1e-3},
                            }
                },
            'losses': #This key specifies the loss function used to compute the loss during training.
                {
                'policy_net':{            
                    'loss':'MSELoss',
                    }
                },
            
            'n_memory':40000,
            'training_stride':5,
            'batch_size':32,
            'saving_stride':100,
            
            'n_episodes_max':10000,
            'n_solving_episodes':40,
            'solving_threshold_min':200,
            'solving_threshold_mean':230,
            
            'discount_factor':0.99,
            }

        parameters2 = self.make_dictionary_keys_lowercase(parameters2)
        parameters2['neural_networks']['target_net'] = {}
        parameters2['neural_networks']['target_net']['layers'] = copy.deepcopy(parameters2['neural_networks']['policy_net']['layers'])

        parameters2['target_net_update_stride'] = 1 
        parameters2['target_net_update_tau'] = 1e-2 

        parameters2['epsilon'] = 1.0 # initial value for epsilon
        parameters2['epsilon_1'] = 0.1 # final value for epsilon
        parameters2['d_epsilon'] = 0.00005 # decrease of epsilon

        parameters = self.merge_dictionaries(dict1=parameters,dict2=parameters2)
        
        self.set_parameters(parameters=parameters)
        self.parameters = copy.deepcopy(parameters)
            
        self.in_training = False
    
    def train(self,environment,
                    verbose=True
                ):

        self.in_training = True
        training_complete = False
        step_counter = 0 
        epoch_counter = 0
        episode_durations = [] 
        episode_returns = [] 
        steps_simulated = [] 
        training_epochs = []  

        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.n_solving_episodes))
            status_progress_string = ( # for outputting status during training
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        
        for n_episode in range(0,self.n_episodes_max):
            state, info = environment.reset()
            current_total_reward = 0.

            for i in itertools.count(): # timesteps of environment
                action = self.act(state=state)
                next_state, reward, terminated, truncated, info = \
                                        environment.step(action)

                step_counter += 1 # increase total steps simulated
                done = terminated or truncated # did the episode end?
                current_total_reward += reward # add current reward to total

                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)
                self.add_memory([torch.tensor(state),
                            action,
                            torch.tensor(next_state),
                            reward,
                            done])

                state = next_state

                if step_counter % self.training_stride == 0:
                    self.run_optimization_step(epoch=epoch_counter) # optimize
                    epoch_counter += 1 # increase count of optimization steps
                
                if done: # if current episode ended
                    episode_durations.append(i + 1)
                    episode_returns.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_epochs.append(epoch_counter)
                    training_complete, min_ret, mean_ret = \
                            self.evaluate_stopping_criterion(list_of_returns=episode_returns)
                    if verbose:
                            if n_episode % 100 == 0 and n_episode > 0:
                                end='\n'
                            else:
                                end='\r'
                            if min_ret > self.solving_threshold_min:
                                if mean_ret > self.solving_threshold_mean:
                                    end='\n'
                            #
                            print(status_progress_string.format(n_episode,
                                    current_total_reward,
                                   min_ret,mean_ret),
                                        end=end)
                    break
            if (n_episode % self.saving_stride == 0) \
                    or training_complete \
                    or n_episode == self.n_episodes_max-1:
                training_results = {'episode_durations':episode_durations,
                            'epsiode_returns':episode_returns,
                            'n_training_epochs':training_epochs,
                            'n_steps_simulated':steps_simulated,
                            'training_completed':False,
                            }

            if training_complete:
                training_results['training_completed'] = True
                break
        
        if not training_complete:
            warning_string = ("Warning: Training was stopped because the "
            "maximum number of episodes, {0}, was reached. But the stopping "
            "criterion has not been met.")
            warnings.warn(warning_string.format(self.n_episodes_max))
        
        self.in_training = False
        
        return training_results

    def make_dictionary_keys_lowercase(self,dictionary):
        output_dictionary = {}
        for key, value in dictionary.items():
            output_dictionary[key.lower()] = value
        return output_dictionary

    def merge_dictionaries(self,dict1,dict2):
        return_dict = copy.deepcopy(dict1)
        
        dict1_keys = return_dict.keys()
        for key, value in dict2.items():
            if key not in dict1_keys:
                return_dict[key] = value
        
        return return_dict

    def set_parameters(self,parameters):
        parameters = self.make_dictionary_keys_lowercase(parameters)

        self.discount_factor = parameters['discount_factor']
        self.n_memory = int(parameters['n_memory'])
        self.memory = memory(self.n_memory)
        self.training_stride = parameters['training_stride']
        self.batch_size = int(parameters['batch_size'])
        self.saving_stride = parameters['saving_stride']
        self.n_episodes_max = parameters['n_episodes_max']
        self.n_solving_episodes = parameters['n_solving_episodes']
        self.solving_threshold_min = parameters['solving_threshold_min']
        self.solving_threshold_mean = parameters['solving_threshold_mean']
        self.target_net_update_stride = parameters['target_net_update_stride']

        self.target_net_update_tau = parameters['target_net_update_tau']
        self.epsilon =  parameters['epsilon']
        self.epsilon_1 = parameters['epsilon_1']
        self.d_epsilon = parameters['d_epsilon']

        self.neural_networks = {}
        for key, value in parameters['neural_networks'].items():
            self.neural_networks[key] = neural_network(value['layers']).to(device)

        self.optimizers = {}
        for key, value in parameters['optimizers'].items():
            self.optimizers[key] = torch.optim.RMSprop(
                        self.neural_networks[key].parameters(),
                            **value['optimizer_args'])
            
        self.losses = {}
        for key, value in parameters['losses'].items():
            self.losses[key] = nn.MSELoss()

    def add_memory(self,memory):
        self.memory.push(*memory)    

    def act(self,state,epsilon=0.):
        
        if self.in_training:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            policy_net = self.neural_networks['policy_net']
            with torch.no_grad():
                policy_net.eval()
                action = policy_net(torch.tensor(state)).argmax(0).item()
                policy_net.train()
                return action
        else:
            return torch.randint(low=0,high=self.n_actions,size=(1,)).item()
        
    def run_optimization_step(self,epoch):
        if len(self.memory) < self.batch_size:
            return
        
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.get_samples_from_memory()
        
        policy_net = self.neural_networks['policy_net']
        target_net = self.neural_networks['target_net']
        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net']
        policy_net.train() 

        LHS = policy_net(state_batch.to(device)).gather(dim=1,index=action_batch.unsqueeze(1))

        Q_next_state = target_net(next_state_batch).max(1)[0].detach()
            
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        RHS = RHS.unsqueeze(1) 

        loss_ = loss(LHS, RHS)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        policy_net.eval() # turn off training mode

        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

        if epoch % self.target_net_update_stride == 0:
            self.soft_update_target_net() # soft update target net
        
    def soft_update_target_net(self):
        params1 = self.neural_networks['policy_net'].named_parameters()
        params2 = self.neural_networks['target_net'].named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.neural_networks['target_net'].load_state_dict(dict_params2)

    def evaluate_stopping_criterion(self,list_of_returns):
        if len(list_of_returns) < self.n_solving_episodes:
            return False, 0., 0.

        recent_returns = np.array(list_of_returns)
        recent_returns = recent_returns[-self.n_solving_episodes:]
        
        minimal_return = np.min(recent_returns)
        mean_return = np.mean(recent_returns)

        if minimal_return > self.solving_threshold_min:
            if mean_return > self.solving_threshold_mean:
                return True, minimal_return, mean_return

        return False, minimal_return, mean_return

    def get_samples_from_memory(self):
        current_transitions = self.memory.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*current_transitions))

        state_batch = torch.cat( [s.unsqueeze(0) for s in batch.state], dim=0)
        next_state_batch = torch.cat([s.unsqueeze(0) for s in batch.next_state],dim=0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.tensor(batch.done).float()

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch