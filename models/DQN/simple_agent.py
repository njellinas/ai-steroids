from collections import deque
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation


class DQNSimpleAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        # Q parameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # DNN parameters
        self.num_hidden = 2
        self.hidden_units = 24
        self.activation = 'elu'

    def set_q_parameters(self, gamma=0.95, epsilon=1.0,
                         epsilon_min=0.01, epsilon_decay=0.995):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def set_dnn_parameters(self, num_hidden=2, hidden_units=24,
                           activation='elu'):
        self.num_hidden = num_hidden
        self.hidden_units = hidden_units
        self.activation = activation

    def build_dnn(self):
        model = Sequential()
        # Input layer
        model.add(InputLayer(input_shape=(1, 4), name='input'))
        # Hidden layers
        for i in range(self.num_hidden):
            model.add(Dense(units=self.hidden_units, input_dim=self.state_dim,
                            kernel_initializer='he_uniform',
                            name='hidden{}'.format(str(i))))
            model.add(Activation(self.activation))
        # Output layer
        model.add(Dense(units=self.action_dim, name='output'))
        # Compile Graph
        model.compile(loss='rmse', optimizer='Adam')
