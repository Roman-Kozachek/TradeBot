import numpy as np
import random
from collections import deque
seed = 17
np.random.seed(seed)

import keras
import tensorflow as tf
from keras import layers

class DQNAgent:
    def __init__(self, observation, action_space_len=7, discount=0.5, replay_memory_size=100000, batch_size=64, alpha=0.001, model=None, update_target_every=10):
        
        self._replay_memory_size = replay_memory_size 
        self._batch_size = batch_size  
        self._discount = discount  
        self._min_replay_memory_size = 1000  
        self._observation_shapes = self._get_observation_shapes(observation)
        self._alpha = alpha  # learning rate
        self._action_space_len = action_space_len
        self._update_target_every = update_target_every
        
        self.model = self.create_model() if model is None else model# Main model 
        self.target_model = keras.models.clone_model(self.model) # Target network
        self.target_model.set_weights(self.model.get_weights())
        
        self._replay_memory = deque(maxlen=self._replay_memory_size)  # Memory deque
        self._update_target_counter = 0
        
    def _get_observation_shapes(self, observation):
        return ((observation["Hour_LSTM"].shape[1], observation["Hour_LSTM"].shape[2]),\
               (observation["M15_LSTM"].shape[1], observation["M15_LSTM"].shape[2]),\
               (observation["M5_LSTM"].shape[1], observation["M5_LSTM"].shape[2]),\
               (observation["M1_LSTM"].shape[1], observation["M1_LSTM"].shape[2]),\
               (observation["State_input"].shape[1]))
    
    def create_model(self):
        # Inputs
        h1_lstm_input = keras.Input(shape=self._observation_shapes[0], name="Hour_LSTM")
        m15_lstm_input = keras.Input(shape=self._observation_shapes[1], name="M15_LSTM")
        m5_lstm_input = keras.Input(shape=self._observation_shapes[2], name="M5_LSTM")
        m1_lstm_input = keras.Input(shape=self._observation_shapes[3], name="M1_LSTM")
        state_input = keras.Input(shape=self._observation_shapes[4], name="State_input")

        # Normalization
        h1_lstm_input_norm = layers.BatchNormalization()(h1_lstm_input)
        m15_lstm_input_norm = layers.BatchNormalization()(m15_lstm_input)
        m5_lstm_input_norm = layers.BatchNormalization()(m5_lstm_input)
        m1_lstm_input_norm = layers.BatchNormalization()(m1_lstm_input)

        # LSTM for sequencial data with Xavier initialization and recurrent regularazer L2
        h1_features = layers.LSTM(64, recurrent_regularizer="l2", kernel_initializer=tf.keras.initializers.GlorotNormal())(h1_lstm_input_norm)
        m15_features = layers.LSTM(64, recurrent_regularizer="l2",  kernel_initializer=tf.keras.initializers.GlorotNormal())(m15_lstm_input_norm)
        m5_features = layers.LSTM(64, recurrent_regularizer="l2",  kernel_initializer=tf.keras.initializers.GlorotNormal())(m5_lstm_input_norm)
        m1_features = layers.LSTM(64, recurrent_regularizer="l2", kernel_initializer=tf.keras.initializers.GlorotNormal())(m1_lstm_input_norm)
        
        # Concatinating processed LSTM outputs
        x = layers.concatenate([h1_features,m15_features, m5_features, m1_features])
        # Feeding concatinated data to a dense layer
        dense_1 = layers.Dense(64, activation="relu", kernel_regularizer="l2", kernel_initializer='random_normal',
    bias_initializer='zeros')(x)
        dense_2 = layers.Dense(64, activation="relu", kernel_regularizer="l2", kernel_initializer='random_normal',
    bias_initializer='zeros')(dense_1)
        x_2 = layers.concatenate([dense_2, state_input])
        # One more dense layer
        dense_3 = layers.Dense(32, activation="relu", kernel_initializer='random_normal',
    bias_initializer='zeros')(x_2)

        # Output
        y = layers.Dense(self._action_space_len, activation="linear", kernel_initializer='random_normal',
    bias_initializer='zeros')(dense_3)

        model = keras.Model(inputs=[h1_lstm_input, m15_lstm_input, m5_lstm_input, m1_lstm_input, state_input],
                           outputs=[y])

        model.compile(
            optimizer=keras.optimizers.Adam(lr=self._alpha),
            loss=[
                keras.losses.Huber(),
            ]
        )
        return model
    
    @tf.function(experimental_relax_shapes=True)
    def predict(self, model, x, batch_size=1):
        """
        A method to quickly predict the action
        """
        with tf.device('/GPU:0'):
            return model(x, batch_size)
    
    def train(self, terminal_state):
        if len(self._replay_memory) < self._min_replay_memory_size:
            return

        minibatch = random.sample(self._replay_memory, self._batch_size)
        current_states = self._create_batch(minibatch, 0)
        current_qs_list = self.predict(self.model, current_states, batch_size=self._batch_size).numpy()

        new_current_states = self._create_batch(minibatch, 3)
        future_qs_list = self.predict(self.target_model, new_current_states, batch_size=self._batch_size).numpy()

        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self._discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            y.append(current_qs)

        self.model.fit(current_states, np.array(y), batch_size=self._batch_size, verbose=0, shuffle=False)
    
        if terminal_state:
            self._update_target_counter += 1
            
        if self._update_target_every == self._update_target_counter:
            self._update_target_counter = 0
            self.target_model.set_weights(self.model.get_weights())
        

    def _create_batch(self, minibatch, i):
        h1 = []
        m15 = []
        m1 = []
        m5 = []
        st = []
        for transition in minibatch:
            h1.append(transition[i]["Hour_LSTM"].squeeze().astype(np.float32))
            m15.append(transition[i]["M15_LSTM"].squeeze().astype(np.float32))
            m5.append(transition[i]["M5_LSTM"].squeeze().astype(np.float32))
            m1.append(transition[i]["M1_LSTM"].squeeze().astype(np.float32))
            st.append(transition[i]["State_input"].squeeze().astype(np.float32))
        return {"Hour_LSTM": np.asarray(h1), "M15_LSTM": np.asarray(m15),\
                 "M5_LSTM": np.asarray(m5), "M1_LSTM": np.asarray(m1),\
                 "State_input": np.asarray(st)}
    
    def reset(self):
        self.model = self.create_model()
        self.target_model = keras.models.clone_model(self.model) # Target network
        self.target_model.set_weights(self.model.get_weights())
    
    def update_replay_memory(self, transition):
        """
        Update memory
        """
        self._replay_memory.append(transition)

    def get_qs(self, state, tg=False):
        """
        Get predicted q-values
        """
        pred = self.predict(self.model if not tg else self.target_model , state)[0].numpy()
        return pred