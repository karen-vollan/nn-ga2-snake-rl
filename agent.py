# sources and support:
# - MAO3

from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json

import torch
import torch.nn as nn
import torch.optim as optim

class Agent():
    """Base class for all agents
    This class extends to the DeepQLearningAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values

    Attributes
    ----------
    _model : Pytorch Graph
        Stores the graph of the DQN model
    _target_net : Pytorch Graph
        Stores the target network graph of the DQN model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)

        # The model loss function - Huber Loss
        self._loss = nn.HuberLoss(reduction="mean")
        self.reset_models()

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model()
        if(self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()

    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        return board.copy()

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : Pytorch Graph, optional
            The graph to use for prediction, model or target network

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        model_outputs = model(board)
        return model_outputs

    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        return (board.astype(np.float32)/4.0).copy()

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves==1, model_outputs.detach().numpy(), -np.inf), axis=1)

    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : Pytorch DQN graph
            DQN model graph
        """
        # Variables the DQN-model need to access for creating the layers
        version = self._version
        board_size = self._board_size
        n_frames = self._n_frames
        n_actions = self._n_actions

        # Creates the DQN model graph with help of nn.Module Class
        # The are also added functions that fit the model to the rest of the code
        class DQN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = self.build_model()

            def optimizer(self, parameters):
                return optim.RMSprop(parameters, lr=0.0005)

            def forward(self, input):
                x = torch.from_numpy(input)
                return self.layers(x)
            
            # This function creates the layers from based on the version model_config information
            def build_model(self):
                # Get the model layer information from the model_config file
                with open('model_config/{:s}.json'.format(version), 'r') as f:
                    m = json.loads(f.read())
                layers = m['model']

                # Help variables for creating the model
                model_layers = []
                in_channels = board_size
                out_channels = None

                # Build the model
                for layer in layers:
                    l = layers[layer]
                    if('Conv2D' in layer):
                        # Get information from model
                        out_channels = l['filters']
                        kernel_size = l['kernel_size']
                        activation = l['activation']

                        # Add convolutional layer
                        model_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=n_frames))

                        # Add activation layer
                        if activation == 'relu': model_layers.append(nn.ReLU())
                        in_channels = out_channels

                    if('Flatten' in layer):
                        model_layers.append(nn.Flatten())

                    if('Dense' in layer):
                        # Get information from model
                        features = l['units']
                        activation = l['activation']
                        # In_features for nn.linear is 5376 for version 17.1. 
                        # It would be better if the in_feature would be calculated in the code, so the code also can work on other versions
                        model_layers.append(nn.Linear(5376, features)) #features
                        if activation == 'relu': model_layers.append(nn.ReLU())
                        in_channels = features
                model_layers.append(nn.Linear(in_channels, n_actions))
                return nn.Sequential(*model_layers)
            
            # This function get the weights from layers that have weight as an attribute
            def get_weights(self):
                weights = []
                idx = 0
                for layer in self.layers:
                    if hasattr(layer, 'weight'):
                        weights.append([idx, layer.weight])
                    idx += 1
                return weights
            
             # This function set the weights to the layers that have weight as an attribute
            def set_weights(self, weights):
                for weight in weights:
                    idx = weight[0]
                    w = weight[1]
                    self.layers[idx].weight = w
                return weights
            
            # Load the weights from disk to memory
            def load_weights(self, file_path):
                self.load_state_dict(torch.load(file_path))
            
            # Save the weight to disk
            def save_weights(self, file_path):
                torch.save(self.state_dict(), file_path)
        return DQN()

    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk with help of functions from DQN-class (the model)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using with help of functions from DQN-class (the model)
        (model saved in h5 format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model.summary())
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net.summary())

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward + 
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if(reward_clip):
            r = np.sign(r)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        # our estimate of expexted future discounted reward
        discounted_reward = r + \
            (self._gamma * np.max(np.where(legal_moves==1, next_model_outputs.detach().numpy(), -np.inf), 
                                  axis = 1)\
                                  .reshape(-1, 1)) * (1-done)
        # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s)
        # we bother only with the difference in reward estimate at the selected action
        target = (1-a)*target.detach().numpy() + a*discounted_reward
        target = torch.from_numpy(target)
        # Get the output
        out = self._get_model_outputs(s, current_model).detach().numpy()
        out = torch.from_numpy(out)
        # Compute the loss (Huber loss)
        loss = self._loss(out, target)

        before = self._model.get_weights()
        # Optimize the model
        loss.requires_grad = True
        optimizer = self._model.optimizer(self._model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return round(loss.item(),5)

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if(self._use_target_net):
            self._target_net.set_weights(self._model.get_weights())