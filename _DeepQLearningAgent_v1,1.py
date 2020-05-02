import time
import copy
import numpy as np
import pandas as pd
import chainer
from chainer import serializers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt

import pdb

class _Backtest:
    
    def __init__(self, data, idx_pairs, sl_percentage,
                 features=90,
                 max_overwater_pos=1,
                 max_underwater_pos=1,
                 commission=1.2e-4,
                 holding_max_t=90):
        """
        idx_pairs: [beginning, end] arrays that the iterator iterates over
        """
        self.data = data
        self.idx_pairs = idx_pairs # index pairs [[beginning, end],...] intervals
        self.sl_percentage = sl_percentage
        self.features = features
        self.max_overwater_pos = max_overwater_pos
        self.max_underwater_pos = max_underwater_pos
        self.commission = commission
        
        ## after t datarows holding a position will be punished
        ## currently works best if you set it as maximum (at the edge of interavl)
        self.holding_max_t = holding_max_t
        
        self.reset()
        
    def reset(self):
        """
        return: will be all zeros, which is the initialization of the
        features = obs array
        """
        self.t_i = 0 # index for idx_pairs
        self.t = self.idx_pairs[self.t_i][0] # chooses beginning of first index pair
        # interval timestep from one interval in idx_pairs to the next one
        self.t_in_pos = 0 # time in position
        self.sl_distance = 0 # stoploss distance
        self.done = False
        self.pnl_r = [] # profit/losses tracking in R's
        self.positions = [] # stores entry price of a position
        self.isLong = True
        self.position_sl = 0
        
        # position value in points (example: _entry 1.5 _current close 2 = 0.5 if long)
        self.position_value = 0
        
        # This creates 90 zero's (if 'features=90 as is standard) in 'self.history'
        # It's the history of close(t)-close(t-1) that will be used as features
        self.history = [0 for _ in range(self.features)]
        
        return [self.position_value] + self.history
    
    def go_short(self):
        self.positions.append(self.data.iloc[self.t, :]['Close'])
        self.set_sl_short()
        self.isLong = False
        
    def go_long(self):
        self.positions.append(self.data.iloc[self.t, :]['Close'])
        self.set_sl_long()
        self.isLong = True
        
    def set_sl_long(self):
        entry = self.entry_price()
        self.position_sl = entry - entry * self.sl_percentage #* (3 / len(self.positions))
        self.sl_distance = entry - self.position_sl
        
    def set_sl_short(self):
        entry = self.entry_price()
        self.position_sl = entry + entry * self.sl_percentage #* (3 / len(self.positions))
        self.sl_distance = self.position_sl - entry
        
    def sl_is_hit_short(self):
        return self.data.iloc[self.t, :]['High'] >= self.position_sl
        
    def sl_is_hit_long(self):
        return self.data.iloc[self.t, :]['Low'] <= self.position_sl
        
    def entry_price(self):
        return np.mean(self.positions)
        
    def register_closed_pos(self, reward_d):
        """
        reward_d: reward distance in points
        """
        # calculate costs of closing this position, which is 2x the
        # commission (in percentage) and 2x - a full "roundturn"
        cost = self.data.iloc[self.t, :]['Close'] * self.commission * 2

        # the reward/risk of the closed position
        rr = round( ( (reward_d - cost) / self.sl_distance ) * 100)

        # add profits in R to pnl_r list
        self.pnl_r.append(rr)

        # no open position anymore
        self.positions = []
        self.t_in_pos = 0
        
        return rr
    
    def step(self, act):
        # act = 0: do nothing, 1: long (close position if short), 2: short (close position if long)
        reward = -20 # negative reward for doing nothing
        
        self.position_value = 0 # initialize position_value to 0
        
        open_pos = len(self.positions)
        
        if open_pos > 0:
            entry = self.entry_price()
            if self.isLong:
                self.position_value = self.data.iloc[self.t, :]['Close'] - entry
            else:
                self.position_value = entry - self.data.iloc[self.t, :]['Close']
        
        # attempting to go long
        if act == 1:
            if open_pos == 0:
                self.go_long()
            else:
                self.t_in_pos += 1
                if self.sl_is_hit_long():
                    reward = self.register_closed_pos(-self.sl_distance)
                else:                                    
                    if not self.isLong:
                        reward = self.register_closed_pos(self.position_value)
                    else:
                        # average (or "add") overwater (in profit) or underwater (in loss)
                        if self.position_value >= 0 and open_pos < self.max_overwater_pos:
                            self.go_long()
                        elif self.position_value < 0 and open_pos < self.max_underwater_pos:
                            self.go_long()
        
        # attempting to go short
        elif act == 2:
            if open_pos == 0:
                self.go_short()
            else:
                self.t_in_pos += 1
                if self.sl_is_hit_short():
                    reward = self.register_closed_pos(-self.sl_distance)
                else:
                    if self.isLong:
                        reward = self.register_closed_pos(self.position_value)
                    else:
                        # average (or "add") overwater (in profit) or underwater (in loss)
                        if self.position_value >= 0 and open_pos < self.max_overwater_pos:
                            self.go_short()
                        elif self.position_value < 0 and open_pos < self.max_underwater_pos:
                            self.go_short()
        
        # punish too long holding times
        if self.t_in_pos > self.holding_max_t:
            reward -= 30
        
        # prepare features for next iteration
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[self.t-1, :]['Close'])
        
        # counts up the time index 'self.t' and the idx_pair index 'self.t_i'
        if self.t < self.idx_pairs[self.t_i][1]: # if inside interval or open position
            self.t += 1
        else:
            self.t_i += 1
            self.t = self.idx_pairs[self.t_i][0]
            if open_pos > 0:
                # closes the position at end of idx_pair intervall (session)
                reward = self.register_closed_pos(self.position_value)

        # obs (observation space = features)
        # standard is, that we also use the current position value as a feature
        # (this will make it possible that the the network learns how to deal with
        # open positions) and the history of the delta close prices
        return [self.position_value] + self.history, reward, self.done # obs, reward, done
        
        
def range_list(stop, start=1, step=1):
    # returns a list of numbers with inputs until when 'stop', 'start' and 'step'
    i = start
    res = []
    while i < stop + step:
        res.append(i)
        i += step
    return res


def slice_df_to_quarters(_df):
    """
    return: pairs of [begin,end] intervals, that can be accessed with
    index >= begin and index < end
    """
    qbm = range_list(12,step=3) # quarter beginning month
    df_slice_pairs = []
    quarter_count = 0
    last_slice_begin = 0
    for t in range(1, len(_df)):
        if _df.iloc[t].name.day < _df.iloc[t-1].name.day and _df.iloc[t].name.month in qbm:
            df_slice_pairs.append([last_slice_begin,t])
            last_slice_begin = t
            quarter_count += 1
    return df_slice_pairs


def plot_loss_reward(total_losses, total_rewards, epochs):
    
    plt.plot(epochs, total_losses)
    
    plt.xlabel('epoch')
    plt.title('loss')
    # plt.savefig("test.png")
    plt.show()
    
    plt.plot(epochs, total_rewards)
    
    plt.xlabel('epoch')
    plt.title('reward')
    # plt.savefig("test.png")
    plt.show()
    
    
def plot_pnl_stats(Q, **kwargs):
    """
    Q: Q-Network with its specific hyperparameters and/or dueling / double q
    mechanism
    **kwargs: Environments with name = environment
    """
    for _name, _env in kwargs.items():
        # the reset function loads the history into this variable
        pobs = _env.reset()
        stepmax = sum([s[1] - s[0] for s in _env.idx_pairs])-1

        for _ in range(stepmax):
        
            pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
            pact = np.argmax(pact.data)
                
            obs, reward, done = _env.step(pact)
        
            pobs = obs
        
        # stat calculations: 
        _equity = [0,]
        profits = []
        losses = []
        trade_count = 0 # trades not 0
        highest = 0
        highest_rel_dd = 0
        for i in _env.pnl_r:
            r_pnl = i/100
            _equity.append(_equity[-1] + r_pnl)
            if r_pnl > 0:
               profits.append(r_pnl)
               trade_count += 1
            else:
                losses.append(r_pnl)
                trade_count += 1
            if _equity[-1] > highest:
                highest = _equity[-1]
                    
            if _equity[-1] < highest:
                dd = highest - _equity[-1]
                if dd > highest_rel_dd:
                    highest_rel_dd = dd
        
        steps = range_list(len(_equity))
        plt.plot(steps, _equity)
        
        plt.xlabel('Trades')
        plt.ylabel('R Multiples')
        plt.title(_name+' Equity')
        # plt.savefig("test.png")
        plt.show()
        
        #print out stats
        hr, avgp, avgl = len(profits)/trade_count, np.mean(profits), abs(np.mean(losses))
        total_profit = sum(_env.pnl_r)/100
        # print(_env.pnl_r) # optional: can print out all pnl_r values in a list
        print(_name,"Stats\n########################\n")
        print('R Profits:',round(total_profit,2))
        print('Trades:',len(profits)+len(losses))
        print('Hitrate:',round(hr,2))
        print('Average Risk/Reward:',round(avgp/avgl,2))
        print('Profit Factor:',round((avgp*hr)/(avgl*(1-hr)),2))
        print('Biggest Rel. DD:',round(highest_rel_dd,2),)
        print('Gain/DD Ratio:',round(total_profit/highest_rel_dd,2),'\n')
        
        
def build_empty_dddqn(env):
    class Q_Network(chainer.Chain):
        # output size = number of states
        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, hidden_size//2),
                fc4 = L.Linear(hidden_size, hidden_size//2),
                state_value = L.Linear(hidden_size//2, 1),
                advantage_value = L.Linear(hidden_size//2, output_size)
            )
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            hs = F.relu(self.fc3(h))
            ha = F.relu(self.fc4(h))
            state_value = self.state_value(hs)
            advantage_value = self.advantage_value(ha)
            advantage_mean = (F.sum(advantage_value, axis=1)/float(self.output_size)).reshape(-1, 1)
            q_value = F.concat([state_value for _ in range(self.output_size)],
                               axis=1) + (advantage_value - F.concat([advantage_mean for _ in range(self.output_size)],
                                                                     axis=1))
            return q_value

        def reset(self):
            self.zerograds()
    
    return Q_Network(input_size=env.features+1, hidden_size=100, output_size=3)


def build_empty_ddqn(env):
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()
    
    return Q_Network(input_size=env.features+1, hidden_size=100, output_size=3)

    
def train_ddqn(env):
    
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()

    Q = Q_Network(input_size=env.features+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

    epoch_num = 50
    step_max = sum([s[1] - s[0] for s in env.idx_pairs])-1
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:

            # select act
            pact = np.random.randint(3)
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        """ <<< DQN -> Double DQN
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        === """
                        indices = np.argmax(q.data, axis=1)
                        maxqs = Q_ast(b_obs).data
                        """ >>> """
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            """ <<< DQN -> Double DQN
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                            === """
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])
                            """ >>> """
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        total_loss += loss.data
                        loss.backward()
                        optimizer.update()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('Epoch: ',epoch+1,' Eps.:',round(epsilon,3),' Total Step: ',total_step,
                  ' Log Rew.: ',round(log_reward,2),' Log Loss: ',round(log_loss,2),
                  ' El. Time: ',round(elapsed_time,0))
            start = time.time()
            
    return Q, total_losses, total_rewards


# Dueling Double DQN

def train_dddqn(env):

    """ <<< Double DQN -> Dueling Double DQN
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()
    === """
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, hidden_size//2),
                fc4 = L.Linear(hidden_size, hidden_size//2),
                state_value = L.Linear(hidden_size//2, 1),
                advantage_value = L.Linear(hidden_size//2, output_size)
            )
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

        def __call__(self, x):
            """
            Build a network that maps state -> action values
            """
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            hs = F.relu(self.fc3(h))
            ha = F.relu(self.fc4(h))
            state_value = self.state_value(hs)
            advantage_value = self.advantage_value(ha)
            advantage_mean = (F.sum(advantage_value, axis=1)/float(self.output_size)).reshape(-1, 1)
            q_value = F.concat([state_value for _ in range(self.output_size)], axis=1) + (advantage_value - F.concat([advantage_mean for _ in range(self.output_size)], axis=1))
            return q_value

        def reset(self):
            self.zerograds()
    """ >>> """

    Q = Q_Network(input_size=env.features+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

    epoch_num = 15
    step_max = sum([s[1] - s[0] for s in env.idx_pairs])-1
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:

            # select act
            pact = np.random.randint(3)
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q = exploit or explore
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    # permutate memory, which means only reall permutate pobs and
                    # obs, because reward, pact and done stay the same
                    shuffled_memory = np.random.permutation(memory)
                    
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        """ <<< DQN -> Double DQN
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        === """
                        indices = np.argmax(q.data, axis=1)
                        maxqs = Q_ast(b_obs).data
                        """ >>> """
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            """ <<< DQN -> Double DQN
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                            === """
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])
                            """ >>> """
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        total_loss += loss.data
                        loss.backward()
                        optimizer.update()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('Epoch: ',epoch+1,' Eps.:',round(epsilon,3),' Total Step: ',total_step,
                  ' Log Rew.: ',round(log_reward,2),' Log Loss: ',round(log_loss,2),
                  ' El. Time: ',round(elapsed_time,0))
            start = time.time()
            
    return Q, total_losses, total_rewards


# DATA PREPARATION
# ################

data = pd.read_csv('GOOG.csv')
print(data.head())
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
print('Data range from:',data.index.min(),"to",data.index.max(),'\n')


# STOPLOSS CALCULATION
# ####################
"""
# Calculate the high to low diffrence in percentages to close price
# to then compute standard deviation from all the 'h2l_ps'
h2l_ps = []
for index, row in data.iterrows():
    h2l_ps.append((row['High'] - row['Low']) / row['Close'])
    
h2l_std = np.std(h2l_ps)
"""
h2l_std = 0.03087
sl_dist = h2l_std * 2


# DATA SEPERATION
# ###############
# slices all data to [beginning, end] interval pairs to quarter intervals
# to make the data more representative
slice_all = slice_df_to_quarters(data)

# shuffles the order of the [begining, end] interval pairs
slice_all = np.random.permutation(slice_all)

# # seperates all slices into two different arrays s_trids and s_teids 
total_slices = len(slice_all)
# here we decide to only leave out 20% holdout data (random)
half_slices = total_slices * 10 // 12
# s_trids: slice of training interval ID's, s_teids: slice of testing interval ID's
s_trids, s_teids  = slice_all[:half_slices], slice_all[half_slices:]


# ENVIRONMENT CREATION
# ####################

env = _Backtest(data,s_trids,sl_dist)


# MODEL CREATION & TESTING
# ########################

# pdb.set_trace() # uncomment this to go through every line of code with debugger
"""
QUICK TUTORIAL:
    if 'MAKE_MODEL' is true, it will train the network and save the model in a file
    and if it is false it will only load the saved network and plot data (this is useful
    if you already have built a model or don't want to train it again)
    
    Note vor v1.1: 'QLearningModel(stock_daily_chart).model' works best.
"""

MAKE_MODEL = False
epochs = range_list(15) # usually 15 epochs are enough (to not overfit)

if MAKE_MODEL:
    Q, total_losses, total_rewards = train_dddqn(env) # Dueling Double Q Learning
    # Q, total_losses, total_rewards = train_ddqn(env) # Double Q Learning
    serializers.save_npz('QLearningModel.model', Q) # saves the model
    # plots losses/rewrads of every epoch
    plot_loss_reward(total_losses, total_rewards, epochs)
else:
    # build an empty network first to copy the loaded Q Object in
    Q = build_empty_dddqn(env) # for Dueling Double Q Learning
    # Q = build_empty_ddqn(env) # for Double Q Learning
    serializers.load_npz('QLearningModel(stock_daily_chart).model', Q) # only loads the model


# PLOTS
# #####

plot_pnl_stats(Q,
               Train = _Backtest(data,s_trids,sl_dist),
               Test = _Backtest(data,s_teids,sl_dist),
               Total =_Backtest(data,slice_all,sl_dist))
