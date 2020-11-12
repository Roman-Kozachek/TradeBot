import numpy as np
import pandas as pd
import datetime as dt
import random
seed = 17
np.random.seed(seed)

from collections import deque
from classes.position import Position

class Environment:
    
    def __init__(self, paths, timesteps, starting_funds=100, max_length=14, start_year=2016, end_year=2019, start_year_test=2019, end_year_test=2020, mondays_only=True, from_start=False, from_start_test=True, end_coef=0.1, end_coef_test=0.5, include_hold_info=True, multiplier=100, stop_loss=None, take_profit=None, action_space=["open.1.5", "open.0.5", "close"], shuffle=True, goal_balance=105):
        """
        In this method we will add some properties to our class as well as set some parameters.
        
        :param starting_funds: amount of starting funds 
        :param max_length: maximum length of a training episode 
        :param paths: a tuple of paths to historical data datasets
        :param timesteps: a tuple of timesteps to return at each step
        """
        self._starting_funds = starting_funds
        self._max_length = max_length 
        self._from_start = from_start
        self._from_start_test = from_start_test
        self._end_coef = end_coef
        self._end_coef_test = end_coef_test
        self._mondays_only = mondays_only
        self._multiplier = multiplier
        self.include_hold_info = include_hold_info
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._goal_balance = goal_balance
        path_to_1h, path_to_15min, path_to_5min, path_to_1min = paths 
        self._h1_timesteps, self._m15_timesteps, self._m5_timesteps, self._m1_timesteps = timesteps 

        self.h1_data = self._get_dataset(path_to_1h)
        self.m15_data = self._get_dataset(path_to_15min)
        self.m5_data = self._get_dataset(path_to_5min)
        self.m1_data = self._get_dataset(path_to_1min)
        
        self.m15_data = self.m15_data.loc[self.m15_data.Date.dt.date < self.h1_data.Date.iloc[-1]]
        self.m5_data = self.m5_data.loc[self.m5_data.Date.dt.date < self.h1_data.Date.iloc[-1]]
        self.m1_data = self.m1_data.loc[self.m1_data.Date.dt.date < self.h1_data.Date.iloc[-1]]
        
        self._max_positions = 1  # number of positions that can be opened at the same time         
        self._leverage = 500  # broker's leverage, a tool to provide small time traders more resources to trade with
        self._commision = 45/(10**6)  # broker's commission per trade
        self._amount_space = np.arange(1, 201)*1000  # range of tradeable currency amounts 
        self._direction = None
        
        # This is our action space. First part indicates general action, second, if present, direction 
        # of trade and third is trading amount in percents
        self.action_space = action_space
        
        self._start_time = pd.to_datetime("06:00", format='%H:%M').time() # Start time of each episode
        self._end_time = pd.to_datetime("20:00", format='%H:%M').time() # End time of each episode
        
        # Create a list of starting indexes in selected time period
        self._start_indexes = self.m1_data.loc[(self.m1_data.Date.dt.time == self._start_time) &\
                                               (self.m1_data.Date.dt.weekday.between(0,5)) &\
                                              (self.m1_data.Date.dt.year.between(start_year,end_year)\
                                              )].index[int(self._h1_timesteps/24):]
        
        self._start_test_indexes = self.m1_data.loc[(self.m1_data.Date.dt.time == self._start_time) &\
                                               (self.m1_data.Date.dt.weekday.between(0,5)) &\
                                              (self.m1_data.Date.dt.year.between(start_year_test,end_year_test)\
                                              )].index[int(self._h1_timesteps/24):]

        if self._mondays_only:
            self._monday_indexes = self.m1_data.loc[(self.m1_data.Date.dt.time == self._start_time) &\
                                                   (self.m1_data.Date.dt.weekday == 0) &\
                                                  (self.m1_data.Date.dt.year.between(start_year,end_year)\
                                                  )].index[int(self._h1_timesteps/24):]
            self._monday_test_indexes = self.m1_data.loc[(self.m1_data.Date.dt.time == self._start_time) &\
                                               (self.m1_data.Date.dt.weekday == 0) &\
                                              (self.m1_data.Date.dt.year.between(start_year_test,end_year_test)\
                                              )].index[int(self._h1_timesteps/24):]
        else:
            self._monday_indexes = self._start_indexes
            self._monday_test_indexes = self._start_test_indexes
        

        
    def _get_dataset(self, path):
        """This method loads dataset from provided path"""
        data = pd.read_csv(path).reset_index(drop=True)
        data.Date = pd.to_datetime(data.Date, format='%Y-%m-%d %H:%M:%S')      
        return data
    
    
    def reset(self, test=False):
        """In this method we will perform actions to reset environment state"""
        # First we'll get a new starting day in our dataset
        if not test:
            if self._from_start:
                self._first_day_index = np.where(self._start_indexes == self._monday_indexes[0])[0][0]
            else:

                self._first_day_index = np.where(self._start_indexes ==\
                                                 random.choice(self._monday_indexes[1:-2]))[0][0]
            self._test = False
        else:
            if self._from_start_test:
                self._first_day_index = np.where(self._start_test_indexes == self._monday_test_indexes[0])[0][0]
            else:

                self._first_day_index = np.where(self._start_test_indexes ==\
                                                 random.choice(self._monday_test_indexes[1:-2]))[0][0]
            self._test = True
        
        self._day_index = 0
        
        self._initial_index = self._start_indexes[self._first_day_index] if not self._test else self._start_test_indexes[self._first_day_index]
        
        self._current_index = self._initial_index
        self._current_state = self.m1_data.iloc[self._current_index]
        self.episode_df = self.m1_data.loc[self.m1_data.Date.dt.date.between(self._current_state.Date.date(), self._current_state.Date.date() + pd.Timedelta(self._max_length, "d")) & self.m1_data.Date.dt.weekday.between(0,5) & (self.m1_data.Date.dt.time.between(self._start_time, self._end_time))][["Date", "Ask"]]  
        self.day_df = self.episode_df.loc[self.episode_df.Date.dt.date == self._current_state.Date.date()]
    
        # Next we'll reset funds, balance and open_positions variables
        self._funds = self._starting_funds
        self._balance = self._funds 
        self._balance_on_open = None
        self._open_positions = {}
        self._cur_open_position = None
        self._open_position_num = 0
        self._hold_counter = 0
        self._open_counter = 0
        
        # Here we'll initialize time queue with historical market data
        self._init_time_deque()

        # Finally we'll reset variables which we send to the agent and return first state
        self._done = False
        self._reward = 0
        self._last_observation = self._get_observation()
        self._info = {"Balance": None, "Funds": None}

        return self._last_observation

    
    def _init_time_deque(self):
        """
        In this method we'll initialize time queue which we'll use to save time when updating environment state
        """
        h1 = self.h1_data.loc[self.h1_data.Date < self._current_state.Date][-self._h1_timesteps-1:-1] # get last values
        self._h1_queue = deque(h1.iloc[:, 1:].to_numpy(), maxlen=self._h1_timesteps) # initiate queue and repeat for every dataset

        m15 = self.m15_data.loc[self.m15_data.Date < self._current_state.Date][-self._m15_timesteps-1:-1]
        self._m15_queue = deque(m15.iloc[:, 1:].to_numpy(), maxlen=self._m15_timesteps)

        m5 = self.m5_data.loc[self.m5_data.Date < self._current_state.Date][-self._m5_timesteps-1:-1]
        self._m5_queue = deque(m5.iloc[:, 1:].to_numpy(), maxlen=self._m5_timesteps)

        m1 = self.m1_data.loc[self.m1_data.Date < self._current_state.Date][-self._m1_timesteps-1:-1]
        self._m1_queue = deque(m1.iloc[:, 1:].to_numpy(), maxlen=self._m1_timesteps)    

        
    def _get_observation(self):
        date_column_index = 0

        if self._current_state.Date.minute == 0:
            row = self.h1_data.iloc[self.h1_data.loc[self.h1_data.Date == self._current_state.Date].index - 1].to_numpy()[0][1:] 
            self._h1_queue.append(row)
            
        if self._current_state.Date.minute % 15 == 0:
            row = self.m15_data.iloc[self.m15_data.loc[self.m15_data.Date == self._current_state.Date].index - 1].to_numpy()[0][1:] 
            self._m15_queue.append(row)

        if self._current_state.Date.minute % 5 == 0:
            row = self.m5_data.iloc[self.m5_data.loc[self.m5_data.Date == self._current_state.Date].index - 1].to_numpy()[0][1:] 
            self._m5_queue.append(row)

        row = self.m1_data.iloc[self.m1_data.loc[self.m1_data.Date == self._current_state.Date].index - 1].to_numpy()[0][1:]   
        self._m1_queue.append(row)

        state = self._get_current_agent_state()

        return {"Hour_LSTM": np.expand_dims(np.array(self._h1_queue), axis=0).astype(np.float32),\
                "M15_LSTM": np.expand_dims(np.array(self._m15_queue), axis=0).astype(np.float32),\
                "M5_LSTM": np.expand_dims(np.array(self._m5_queue), axis=0).astype(np.float32),\
                "M1_LSTM": np.expand_dims(np.array(self._m1_queue), axis=0).astype(np.float32),\
                "State_input": np.expand_dims(np.array(state), axis=0).astype(np.float32)}

    def _init_time_deque_test(self):
        """
        In this method we'll initialize time queue which we'll use to save time when updating environment state
        """
        h1 = self.h1_data.loc[self.h1_data.Date < self._current_state.Date][-self._h1_timesteps-1:-1] # get last values
        self._h1_queue = deque(h1.to_numpy(), maxlen=self._h1_timesteps) # initiate queue and repeat for every dataset

        m15 = self.m15_data.loc[self.m15_data.Date < self._current_state.Date][-self._m15_timesteps-1:-1]
        self._m15_queue = deque(m15.to_numpy(), maxlen=self._m15_timesteps)

        m5 = self.m5_data.loc[self.m5_data.Date < self._current_state.Date][-self._m5_timesteps-1:-1]
        self._m5_queue = deque(m5.to_numpy(), maxlen=self._m5_timesteps)

        m1 = self.m1_data.loc[self.m1_data.Date < self._current_state.Date][-self._m1_timesteps-1:-1]
        self._m1_queue = deque(m1.to_numpy(), maxlen=self._m1_timesteps)    
    
    
    def _get_observation_test(self):
        date_column_index = 0

        if self._current_state.Date.minute == 0:
            row = self.h1_data.iloc[self.h1_data.loc[self.h1_data.Date == self._current_state.Date].index - 1].to_numpy()[0] 
            self._h1_queue.append(row)
            
        if self._current_state.Date.minute % 15 == 0:
            row = self.m15_data.iloc[self.m15_data.loc[self.m15_data.Date == self._current_state.Date].index - 1].to_numpy()[0]
            self._m15_queue.append(row)

        if self._current_state.Date.minute % 5 == 0:
            row = self.m5_data.iloc[self.m5_data.loc[self.m5_data.Date == self._current_state.Date].index - 1].to_numpy()[0] 
            self._m5_queue.append(row)

        row = self.m1_data.iloc[self.m1_data.loc[self.m1_data.Date == self._current_state.Date].index - 1].to_numpy()[0]   
        self._m1_queue.append(row)

        state = self._get_current_agent_state()

        return {"Hour_LSTM": self._h1_queue,\
                "M15_LSTM": self._m15_queue,\
                "M5_LSTM": self._m5_queue,\
                "M1_LSTM": self._m1_queue,\
                "State_input": state}
    
    def step(self, action):
    
        action = self.action_space[action]
        action = action.split(".")
        closing_profit = None
        reward = 0
        # Taking action
        if action[0]=="open":
            self._direction = int(action[1])
            reward = self._close_positions(not self._direction)
            if self._open_position_num < self._max_positions:
                self._balance_on_open = self._balance
                usd_cost = self._balance*(0.01*int(action[2]))
                exchange_rate = self._current_state.Ask if self._direction==1 else self._current_state.Bid
                cost_space = self._amount_space*exchange_rate/self._leverage
                usd_cost = cost_space[(np.abs(cost_space - usd_cost)).argmin()]
                 

                usd_amount = round(usd_cost*self._leverage, 2)
                eur_amount = self._amount_space[np.where(cost_space == usd_cost)][0]
                transaction_cost = round(self._commision*usd_amount, 2)
                self._funds = round(self._funds - usd_cost - transaction_cost, 2)
                pos = Position(self._current_state, self._direction, exchange_rate,\
                                                                           transaction_cost, usd_cost, usd_amount, \
                                                                           eur_amount, self._leverage)
                self._open_positions[len(self._open_positions)] = pos
                self._cur_open_position = pos
            self._hold_counter = 0
            
        elif action[0]=="close":
            reward = self._close_positions()
            self._direction = None
            
        elif action[0]=="hold":
            if self._open_position_num>0:
                self._open_counter +=1
            else:
                self._hold_counter +=1
            reward = 0

        
        self._update_state(reward)
        
        return self._last_observation, self._reward, self._done, self._info
    
        
    def _close_positions(self, direction=None):
        i = 0 
        for key, position in self._open_positions.items():
            if position.open and ((position.direction == direction) if direction is not None else True):
                i += 1
                self._close_position(position)
                self._open_position_num -= 1
        
        if i>0:
            self._open_counter = 0
            self._cur_open_position = None
            reward = (self._balance - self._balance_on_open)/self._balance_on_open*self._multiplier
        else:
            reward = 0
                    
        return reward
    
    def _close_position(self, position):
        self._funds = round(self._funds + position.get_value() - position.transaction_cost, 2)
        position.close(self._current_state)
    
    def _update_state(self, reward): 
        """
        Updates the state of the environment
        """
        old_balance = self._balance  # Save old balance for reward calculation
        
        
        if self._current_state.Date.time() >= self._end_time:  
            self._close_positions()
            self._balance = self._funds

            self._day_index +=1
            if not self._test:
                self._current_index = self._start_indexes[self._first_day_index + self._day_index]
            else:
                self._current_index = self._start_test_indexes[self._first_day_index + self._day_index]
            self._current_state = self.m1_data.iloc[self._current_index]
            self.day_df = self.episode_df.loc[self.episode_df.Date.dt.date == self._current_state.Date.date()]
            self._init_time_deque()
            new_day = True
            self._hold_counter = 0
            self._open_counter = 0
        else:
            self._current_index += 1
            self._current_state = self.m1_data.iloc[self._current_index]
            new_day = False

        self._update_open_positions()

        self._balance = round(self._funds + self._get_open_positions_value(), 2)
        if not self._test:
            if self._balance <= self._starting_funds*self._end_coef or self._funds <= self._starting_funds*self._end_coef:
                self._done = True
        else:
            if self._balance <= self._starting_funds*self._end_coef_test or self._funds <= self._starting_funds*self._end_coef_test:
                self._done = True
        if self._day_index >= self._max_length:
            self._done = True

        if not self._done:
            self._reward = reward
        else:
            result = self._balance - self._starting_funds
            self._reward = result/self._starting_funds*self._multiplier if result<(self._goal_balance - self._starting_funds) else 1*self._multiplier
        self._last_observation = self._get_observation()
        self._info["Balance"]  = round(self._balance, 2)
        self._info["Funds"] =  round(self._funds, 2)
        self._info["Open_positions"] = sum(1 for pos in self._open_positions.values() if pos.open)
        self._info["ND"] = new_day

    def _update_open_positions(self):
        """
        Updates the state of positions
        """
        self._open_position_num = 0
        for position in self._open_positions.values():
            if position.open:
                position._update_position(self._current_state)
                if self._stop_loss is not None:
                    if (position.cost_usd + position.current_profit) < ((1 - self._stop_loss)*position.cost_usd):
                        self._close_position(position)
                if self._take_profit is not None:
                    if (position.cost_usd + position.current_profit) > ((1 + self._take_profit)*position.cost_usd):
                        self._close_position(position)
                else:
                    self._open_position_num +=1

                
    def _get_open_positions_value(self):
        """
        Calculates value of currently open positions
        """
        value = 0
        for position in self._open_positions.values():
            if position.open:
                value += position.get_value()   
        return value

    def _get_open_positions_profit(self):
        """
        Calculates profit of currently open positions
        """
        profit = 0
        for position in self._open_positions.values():
            if position.open:
                profit += position.get_profit()   

        return profit

    
    def _get_current_agent_state(self):
        """
        Returns information about agent's state in the environment
        """
        open_ask = sum([1 for position in list(self._open_positions.values()) if position.open and position.direction==1])
        open_bid = sum([1 for position in list(self._open_positions.values()) if position.open and position.direction==0])               
        info = [open_ask, open_bid, self._get_open_positions_profit()/self._balance*self._multiplier, (self._balance-self._starting_funds)]
        if self.include_hold_info:
            info.extend([self._open_counter, self._hold_counter])
        return tuple(info)