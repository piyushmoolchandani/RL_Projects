from keras import layers, models, optimizers
from keras import backend as K
from .replay_buffer import ReplayBuffer

class Actor():
    
    def __init__(self, state_size, action_size, action_low, action_high):
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()
        
    def build_model(self):
        
        states = layers.Input(shape = (self.state_size,), name = 'states')
        net = layers.Dense(units = 16, activation = 'relu')(states)
        net = layers.Dense(units = 64, activation = 'relu')(net)
        net = layers.Dense(units = 32, activation = 'relu')(net)
        net = layers.Dense(units = 8, activation = 'relu')(net)
        
        actions_fractions = layers.Dense(units = self.action_size, activation = 'sigmoid', name = 'action')(net)
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name = 'actions')(actions_fractions)
        
        self.model = models.Model(inputs = states, outputs = actions)
        
        action_gradients = layers.Input(shape = (self.action_size, ))
        loss = K.mean(-action_gradients * actions)
        
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params = self.model.trainable_weights, loss = loss)
        
        self.train_functions = K.function(
            inputs = [self.model.input, action_gradients, K.learning_phase()], 
            outputs = [],
            updates = updates_op)
        
class Critic:

    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.Dense(units=16, activation='relu')(net_states)

        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions = layers.Dense(units=16, activation='relu')(net_actions)

        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        
        Q_values = layers.Dense(units=1, name='q_values')(net)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        action_gradients = K.gradients(Q_values, actions)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)