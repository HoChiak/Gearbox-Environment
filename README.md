# Introduduction

This repository shows the implementation of a **Custom Environment** in [GYM](https://gym.openai.com/) using [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html).


In Detail: This Repository is a [GYM](https://gym.openai.com/) Implementation for [Gearbox](https://github.com/HoChiak/Gearbox/). The Gearbox-RePo simulates the vibration behaviour of a gearbox under degradation. In terms of Reinforcement Learning the goal is to decrease the degradation to a minimum. Actions are taken by applying an adapted torque input strategy.

Further Aspects of this Repository are:
* Custom Policy


The intersection between this Repository and the [Gearbox](https://github.com/HoChiak/Gearbox/) Repository is shown in the following picture.

<img src="https://https://github.com/HoChiak/Gearbox-Environment/blob/master/GearboxEnvironment.png" width="60%">

# Setting up

Install the following packages and their dependencies:

`pip install tensorflow==1.14.0`

`pip install gym`

`pip install numpy >= 1.17`

`pip install keras-rl`

(The exact Anaconda Environment is defined in AnacondaRepoExplicit.txt)


Install `stable-baselines` with respect to [Guideline](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites)


Building an GYM Environment based on the original toolbox follows the explanations given in [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) and [[1]](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/), [[2]](https://gym.openai.com/), [[3]](https://www.novatec-gmbh.de/en/blog/creating-a-gym-environment/), [[4]](https://ai-mrkogao.github.io/reinforcement%20learning/openaigymtutorial/), [[5]](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa) [[6]](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)




# Versions

The following versions have been available:

**Gearbox-Environment | Branch: 0.1 | [Gearbox Branch:](https://github.com/HoChiak/Gearbox/tree/0.6.1) 0.6.1** <u>Current Version</u>


---
# Import Gearbox and define necessary attributes
---

Import [Gearbox](https://github.com/HoChiak/Gearbox/) and associates


```python
from gearbox import Gearbox
import gearbox_functions as gf
from GearboxParams import *
```


```python
flag_rlalgor = 'PPO2' # 'PPO2', 'DQN'
flag_stblbsln = 'common' # 'common' for PPO, | 'deepq' for DQN
total_timesteps = int(2e5)
nolc_step = 5e5
gamma = 0#.99
learning_rate = 0.00025
tensorboard_log = './%s/' % (tag+'_'+flag_rlalgor)
no_conv_layer = 1
stride = 10
n_filters = 1
n_hidden = 32
```

Gearbox Input Params


```python
rotational_frequency_in = 1300/60*41/21 # U/s | float
number_of_load_cycle = 0 # | Must be float in .3f 
sample_interval = 0.25 # s | float
sample_rate =int(51200/2)#/4 # Hz | float 4
seed = 8
```

Get Initial Torque


```python
sample_time = gf.get_sample_time_torque(rotational_frequency_in, sample_rate, GearIn['no_teeth'], GearOut['no_teeth'])
initial_torque = np.ones(sample_time.shape) * 200 # Nm | array
```

Initialize a new Instance of Gearbox


```python
gearbox = Gearbox(rotational_frequency_in,
                  sample_interval, sample_rate,
                  # Vibration Arguments
                  GearIn, GearOut,
                  Bearing1, Bearing2, Bearing3, Bearing4,
                  # Degradation Arguments
                  Deg_GearIn, Deg_GearOut,
                  Deg_Bearing1, Deg_Bearing2, Deg_Bearing3, Deg_Bearing4,
                  # Shared Arguments
                  seed=seed,
                  verbose=1, # 0: no output of "load cycle #### done"
                  fixed_start=True,
                  GearDegVibDictIn=GearDegVibDictIn,
                  GearDegVibDictOut=GearDegVibDictOut)
```

---
# Define Interpreter, Vibration2State and Action2Torque
---

Besides the agent there are three other modules (Interpreter, Vibration2State, Action2Torque) which will be defined in the following.

<img src="https://https://github.com/HoChiak/Gearbox-Environment/blob/master/GearboxEnvironment.png" width="60%">

Load Modules


```python
# Build In
import os
from copy import deepcopy as dc
import sys
from datetime import datetime

# Third Party
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Gym and Stable Baselines
import gym
from gym import error, spaces, utils
# from gym.utils import seeding
from stable_baselines.common.env_checker import check_env
import tensorflow as tf

```

## Interpreter


```python
interpreter_choice = 'step1' 


def interpreter(vibrations, nolc, interpreter_parameters):
    """
    Interpreter for Gearbox Environment.
    Type of interpreter function(s) can be choosen by:
    interpreter_parameters['interpreter_choice']
    More than one option can be specified (summed up)
    Options implemented:
    ----
    step1:  Gives constant reward by given number of 
            load cycle (nolc)
    ----
    Argument 'interpreter_parameters' is input 
    and output -> used for recursive calculations etc.
    ---
    
    """
    # Placeholder
    rewards = []
    if 'step1' in interpreter_parameters['interpreter_choice']:
        # ------ Get same normed reward each step
        reward1 = (nolc-interpreter_parameters['prev_values']['nolc']) / 1e6
        reward1 = float(reward1)
        # Append
        rewards.append(reward1)
        # To keep return unchanged argument metric must be defined
        metric = None
    reward = float(sum(rewards)) # ensure reward is scalar float
    return(reward, {'prev_values': {'nolc': nolc}, 'interpreter_choice': interpreter_parameters['interpreter_choice']})

#-------------------------------------------------------------------------
# Parameters for kwargs
interpreter_parameters = {'prev_values': {'nolc': 0
                                         },
                          'interpreter_choice': interpreter_choice}
```

## Vibrations2Observations


```python
def vibrations2observations(vibrations, observation_parameters):
    """
    Shaping Vibrations into Observations for Gearbox Environment.
    Calculation is done as follows
    ----
    1. Standardize features by removing the mean and scaling to 
    unit variance - using recursive mean and variance calculations
    (limiting change from step to step for more stable results)
    2. Limit observations to: -5 < obs < 5
    ----
    Argument 'observation_parameters' is input 
    and output -> used for recursive calculations etc.
    ---
    """
    # Get current recursive mean and variance
    if observation_parameters['prev_values']['mean'] is np.nan:
        mean = np.mean(vibrations)
        var = np.var(vibrations)
    else:
        # Recursive Averaging by given weight
        weight_new = 2
        mean = (( observation_parameters['prev_values']['mean'] * observation_parameters['prev_values']['n'] + np.mean(vibrations) * weight_new) / (observation_parameters['prev_values']['n'] + 1 * weight_new))
        var = (( observation_parameters['prev_values']['var'] * observation_parameters['prev_values']['n'] + np.var(vibrations) * weight_new) / (observation_parameters['prev_values']['n'] + 1 * weight_new))
    obs = (vibrations - mean) / np.power(var, 0.5)
    n = observation_parameters['prev_values']['n'] + 1
    obs = obs.reshape(-1, 1)
    obs[obs < -5] = -5
    obs[obs > 5] = 5           
    return(obs, {'prev_values': {'mean': mean, 'var': var, 'n': n}})

#-------------------------------------------------------------------------
# Parameters for kwargs
observation_parameters = {'prev_values': {'mean': np.nan, 'var': np.nan, 'n': 0}}

```

Define Observation Space


```python
observation_space = spaces.Box(low=-5, high=5,
                        shape=(np.floor(sample_interval*sample_rate).astype(np.int32), 1),
                        dtype= np.float32)
```

## Action2Torque

The following functions take the Agents output (integer determining the tooth to reduce torque at) and output a torque signal with respect to the reducement. Further explaination will be updated.


```python
no_actions = GearIn['no_teeth'] + 1
rotational_frequency_in = 1300/60*41/21
sample_rate = 51200/4

def get_binary_load_dict(no_teeth, reduce_at_tooth=None, reduce_to_torque=None, standard_torque=200):
    """
    """
    balance_torque = standard_torque + (standard_torque - reduce_to_torque) / (no_teeth - 1)
    load_dict = {'%i' % (idx): balance_torque for idx in range(1, no_teeth+1) if idx!=reduce_at_tooth}
    load_dict['%i' % (reduce_at_tooth)] = reduce_to_torque
    return(load_dict)


def repeat2no_values(vector, no_values):
    """
    Repeat the given vector as many times needed,
    to create a repeat_vector of given number of
    values (no_values)
    """
    # Calculate number of repetitions
    no_values_vector = vector.shape[0]
    repetitions = np.ceil((no_values / no_values_vector))
    repetitions = int(repetitions) #dtype decl. not working
    # Repeat Vetor
    repeat_vector = np.tile(vector, repetitions)
    # Trim to final length
    repeat_vector = np.delete(repeat_vector,
                              np.s_[no_values:], axis=0)
    return(repeat_vector)


def get_cids(time, time_shift, time_start=0, id_start=0):
    """
    Shift a given signal by a given time shift.
    """
    # Shift signal for each gear
    ti, tv = id_start, time_start
    #shifted_signal = np.zeros((time.shape[0], 1))
    cid_list = list()
    while tv < (max(time)+time_shift):
        # Add current center id to list
        cid_list.append(ti)
        # Get new shift arguments
        tv += time_shift
        ti = np.argmin(np.abs(time - tv))
    # Remove first zero axis
    #shifted_signal = np.delete(shifted_signal, 0, 1)
    return(cid_list)


def torque_from_dict(load_dict, rotational_frequency, sample_time, get_cids=get_cids):
    """
    Method to determine an aquivalent load for each tooth.
    Returns a dictionary containing a list of mean loads
    per tooth. E.g.
    '1': [155, 177, 169,....]
    '2': [196, 155, 169,....]
    '3' ...
    ....
    """
    no_teeth = len(load_dict)
    time2tooth = (1 / rotational_frequency) / no_teeth
    teeth_cid_list = get_cids(time=sample_time, time_shift=time2tooth,
                                  time_start=0, id_start=0)
    teeth_numbering = np.arange(1, no_teeth+0.1, 1, dtype=np.int32)
    teeth_no_list = repeat2no_values(teeth_numbering, no_values=len(teeth_cid_list))
    # Get Tooth Center IDs
    ids_array = np.array(teeth_cid_list)
    ids_array = ids_array.reshape(-1, 1)
    # Get distance between 2 tooth in no ids
    dist_ids = ids_array[1] - ids_array[0]
    # Take half
    dist_ids = dist_ids / 2
    # Get upper and lower bound
    #ids_low = np.floor(ids_array - dist_ids)
    ids_up = np.floor(ids_array + dist_ids)
    # Correct for highest and lowest possible id
    #ids_low[ids_low < 0] = 0
    ids_up[ids_up > (sample_time.size -1)] = sample_time.size
    ids_up = ids_up.tolist()
    # Add to one array
    #ids_bounds = np.concatenate([ids_low, ids_up], axis=1).astype(dtype=np.int32)
    # Get empty array
    torque = np.zeros(sample_time.shape)
    # Iterate over torque and get mean value of load per tooth and load cycle
    id_low = int(0)
    for idx, id_up in enumerate(ids_up):
        torque[id_low:int(id_up[0])] = load_dict[str(teeth_no_list[idx])]
        id_low = int(id_up[0])
    return(torque)

# ------------------------------------------------
# Change the following paragraph for different learning approaches
# ------------------------------------------------
def action2torque(action, initial_torque,
                  action_parameters):
    """
    Takes an Action (integer) and outputs an torque
    signal
    Every other used function must be passed by 
    action parameters!
    """
    reduce_at_tooth = int(action)
    if reduce_at_tooth == 0:
        """
        Do nothing and return initial torque
        """
        return(initial_torque)
    else:
        """
        Reduce at tooth given by action integer
        """
        get_binary_load_dict = action_parameters['get_binary_load_dict']
        load_dict = get_binary_load_dict(action_parameters['no_actions'] - 1,
                                         reduce_at_tooth=reduce_at_tooth,
                                         reduce_to_torque=190,
                                         standard_torque=200)
        get_sample_time_torque = action_parameters['get_sample_time_torque']
        sample_time = get_sample_time_torque(action_parameters['rotational_frequency'],
                                            action_parameters['sample_rate'],
                                            action_parameters['GearIn_teeth'],
                                            action_parameters['GearOut_teeth'])
        torque_from_dict = action_parameters['torque_from_dict']
        get_cids = action_parameters['get_cids']
        torque = torque_from_dict(load_dict, action_parameters['rotational_frequency'],
                                    sample_time, get_cids=get_cids)
        return(np.array(torque).astype(np.float64))

#-------------------------------------------------------------------------
# Parameters for kwargs
action_parameters = {'no_actions': no_actions,
                     'rotational_frequency': rotational_frequency_in,
                     'sample_rate': sample_rate,
                     'GearIn_teeth': GearIn['no_teeth'],
                     'GearOut_teeth': GearOut['no_teeth'],
                     'get_binary_load_dict': get_binary_load_dict,
                     'get_sample_time_torque': gf.get_sample_time_torque,
                     'torque_from_dict': torque_from_dict,
                     'get_cids': get_cids
                    }



```

Define Action Space


```python
action_space = spaces.Discrete(no_actions) # Add one, if action=0 do nothing
```

---
# Gearbox-Environment OpenAI-Style (Gym)
---

## PseudoCode 


**Define a class 'GearboxBaseEnv' and initialize by giving gearbox, interpreter, action2torque, vibrations2observations, etc.**


```
class GearboxBaseEnv(gym.Env):

    def __init__(self, gearbox, initial_torque, *args, **kwargs):
        """ Initialize the environment with specific settings. Settings include: """
        # ------------------------------------------------
        self.gearbox = gearbox
        ...
        self.action_space = action_space
        self.observation_space = observation_space
        # ------------------------------------------------
```



**Define step() method taking 'action' as input and returning 'observations, reward, done and info'**

```
    def step(self, action):
        """ Performing a step includig: Take Action, Get Reward, Get Observations and define end of episode """
        # ------------------------------------------------
        ...
        self.gearbox.set(self.nolc, self.torque)
        self.vibrations = self.gearbox.run(self.nolc, output=True)
        self.obs = self.vibrations2observations(self.vibrations, self.kwargs['observation_parameters'])
        ...
        self.reward, self.kwargs['interpreter_parameters'] = self.interpreter([self.vibrations], self.nolc, self.kwargs['interpreter_parameters'])
        # ------------------------------------------------
        return(self.obs, self.reward, self.done, self.info)
```



**Define reset() method resetting environment after episode**
```
    def reset(self):
        """ Reinitialize Environment and reset initial settings """
        # ------------------------------------------------
        self.action = 0 # apply initial_torque
        self.vibrations = self.gearbox.run(self.nolc, output=True)
        self.obs = self.vibrations2observations(self.vibrations, self.kwargs['observation_parameters'])
        ...
        # ------------------------------------------------
        return(self.obs)
```



**Define render() method to ouput some information**

```
    def render(self, mode='ansi', close=False):
        """ Renders the environment. """
        # ------------------------------------------------
        print(txt_ansi, end="\r")
        ...
```



**Define close() method to close**

```
    def close(self):
        pass
```



**Define other methods necessary for gearbox**


```
    def nextseed(self):
        """
        Method to get new seed for next episode,
        different than previous seed.
        """
        seed = dc(self.seed)
        seed += np.random.randint(1, high=10, size=1, dtype=np.int32)[0]
        if seed > 2**16:
            seed = seed - 2**16
        return(seed)


    def check_stop_criteria(self, statei, criteria):
        """
        Method to check if criteria is reached in statei.
        Currently it checks if any gear pitting is >= criteria,
        e.g. 4 %.
        Returns False if stop criteria is not reached
        Returns True if stop criteria is reached
        """
        for key in statei.keys():
            if statei[key] is not None:
                if (statei[key] >= criteria).to_numpy().any():
                    return(False)
        return(True)

    def startpoint_detection(self):
        if self.gearbox.Degradation.GearIn_Degradation.state0 is not None:
            gearin_n0_min = min(self.gearbox.Degradation.GearIn_Degradation.state0['n0'])
        else:
            gearin_n0_min = np.inf
        if self.gearbox.Degradation.GearOut_Degradation.state0 is not None:
            gearout_n0_min = min(self.gearbox.Degradation.GearOut_Degradation.state0['n0'])
        else:
            gearout_n0_min = np.inf

        n0_min = min(gearin_n0_min, gearout_n0_min)
        n0_min = max(n0_min, 0)
        remainder = n0_min % self.nolc_step
        startpoint = n0_min - remainder + self.nolc_step
        return(startpoint)
```

## Real Code (Non-PseudoCode) formulation:


```python
class GearboxBaseEnv(gym.Env):
    metadata = {'render.modes': ['ansi']} # should contain all available render modes

    def __init__(self, gearbox, initial_torque,
                 nolc_step,
                 interpreter=None,
                 vibrations2observations=None,
                 action2torque=None,
                 observation_space=None,
                 action_space=None,
                 stop_criteria=4.0, seed=None,
                 render_in_step=False,
                 warn_limit=None,
                 verbose=0,
                 render_mode='ansi2',
                 dense=True, # if False reward will only given when self.done = True
                 **kwargs):
        """
        Initialize the environment with specific settings. Settings include:
        env: gearbox environment
        done: is True if end of episode is reached
        counter: counting steps
        action and observation space
        """
        assert interpreter is not None, 'Function interpreter() must be given'
        assert vibrations2observations is not None, 'Function vibrations2observations() must be given'
        assert action2torque is not None, 'Function action2torque() must be given'
        assert action_space is not None, 'Gym Spaces action_space must be given'
        assert observation_space is not None, 'Gym Spaces observation_space must be given'
        # ------------------------------------------------
        # Init Dummy Environment --> first real initialization is done in reset()
        # ------------------------------------------------
        self.gearbox = gearbox
        self.initial_torque = np.array(initial_torque).astype(np.float64)
        # Initialize here and only reinitialize in reset()
        self.gearbox.initialize(self.initial_torque)
        self.nolc_step = nolc_step
        self.interpreter = interpreter
        self.vibrations2observations = vibrations2observations
        self.action2torque = action2torque
        self.stop_criteria = stop_criteria
        self.seed = seed
        self.render_in_step = render_in_step
        self.warn_limit = warn_limit
        self.verbose = verbose
        self.render_mode = render_mode
        self.dense = dense
        self.kwargs = kwargs
        self.kwargs_init = dc(kwargs)
        # Until done is False -> keeprunning
        self.done = False
        self.counter = 0
        self.episode = -1
        self.reward = np.nan
        self.infos = []
        self.render_len_max = 0
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        # ------------------------------------------------
        # Change the following paragraph for different learning approaches
        # ------------------------------------------------
        self.action_space = action_space
        self.observation_space = observation_space
        # ------------------------------------------------

## STEP defined in a Way to prevend recursive reward!!!
    def step(self, action):
        """
        Performing a step includig: Take Action, Get Reward,
        Get Observations and define end of episode

        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.action = action
        # ------------------------------------------------
        # Take Action at previous nolc
        # ------------------------------------------------
        self.torque = self.action2torque(action, self.initial_torque,
                                         self.kwargs['action_parameters'])
        self.gearbox.set(self.nolc, self.torque)
        # ------------------------------------------------
        # Go to current nolc
        # ------------------------------------------------
        self.nolc += self.nolc_step
        self.counter += 1
        # ------------------------------------------------
        # Get Observations
        # ------------------------------------------------
        self.vibrations = self.gearbox.run(self.nolc, output=True)
        self.obs, self.kwargs['observation_parameters'] = self.vibrations2observations(self.vibrations, self.kwargs['observation_parameters'])
        # ------------------------------------------------
        # Check Done
        # ------------------------------------------------
        self.done = not(self.check_stop_criteria(self.gearbox.ga_statei[-1], self.stop_criteria))
        # ------------------------------------------------
        # Warn if nolc limit is given
        # ------------------------------------------------
        if self.warn_limit is not None:
            if self.nolc >= self.warn_limit:
                warnings.warn('The current load cycle exceeded the warning limit, episode will be ended.', UserWarning)
                self.done = True
                self.gearbox.Degradation.summary_degradation()
        # ------------------------------------------------
        # Get Rewards
        # Change the following paragraph for different learning approaches
        # ------------------------------------------------
        if (not(self.dense) and not(self.done)):
            self.reward = 0
        else:
            self.reward, self.kwargs['interpreter_parameters'] = self.interpreter([self.vibrations], self.nolc, self.kwargs['interpreter_parameters'])
        # ------------------------------------------------
        # Get Info
        # ------------------------------------------------
        if (self.verbose==1 or (self.verbose==2 and self.done)):
            self.info['counter'] = self.counter
            self.info['episode'] = self.episode
            self.info['nolc'] = self.nolc
            # self.info['observations'] = self.obs
            self.info['action'] = action
            self.info['reward'] = self.reward
            try:
                self.info['prev_values'] = self.kwargs['interpreter_parameters']['prev_values']
            except:
                pass
            # self.info['history'] = [-1]
            self.infos.append(dc(self.info))
        # ------------------------------------------------
        # Render if render_in_step is True
        # ------------------------------------------------
        if self.render_in_step:
            self.render()
        # ------------------------------------------------
        # Return Observations, Reward, Done, Info
        # ------------------------------------------------
        return(self.obs, self.reward, self.done, self.info)


    def reset(self):
        """
        REinitialize Environment and reset initial settings
        """
        # ------------------------------------------------
        # Change the following paragraph for different learning approaches
        # ------------------------------------------------
        self.action = 0 # apply initial_torque
        # ------------------------------------------------
        # Init Environment
        # ------------------------------------------------
        self.gearbox.ga_seed = self.seed
        # REinitialize() only resets Degradation (~50times faster than initialize())
        self.gearbox.reinitialize(self.initial_torque)
        # Until done is False -> keeprunning
        self.done = not(self.check_stop_criteria(self.gearbox.ga_statei[-1], self.stop_criteria))
        self.counter = 0
        self.episode += 1
        # Get new seeding
        self.seed = self.nextseed()
        self.kwargs = dc(self.kwargs_init)
        # ------------------------------------------------
        # Change the following paragraph for different learning approaches
        # ------------------------------------------------
        self.nolc = self.startpoint_detection()
        self.vibrations = self.gearbox.run(self.nolc, output=True)
        self.obs, self.kwargs['observation_parameters'] = self.vibrations2observations(self.vibrations, self.kwargs['observation_parameters'])
        # ------------------------------------------------
        # Get Info
        # ------------------------------------------------
        self.info = {}
        if (self.verbose==1 or (self.verbose==2 and self.done)):
            self.info['counter'] = self.counter
            self.info['episode'] = self.episode
            self.info['nolc'] = self.nolc
            # self.info['observations'] = self.obs
            self.info['action'] = None
            self.info['reward'] = None
            try:
                self.info['prev_values'] = self.kwargs['interpreter_parameters']['prev_values']
            except:
                pass
            # self.info['history'] = [-1]
            self.infos.append(dc(self.info))
        # ------------------------------------------------
        # Return Observations
        # ------------------------------------------------
        return(self.obs)

    def render(self, mode='ansi', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        mode=self.render_mode

        if mode == 'rgb_array': # return RGB frame suitable for video
            pass
        elif mode == 'human': # pop up a window and render
            pass
        elif mode == 'ansi1': # return terminal-style text representation
            # ------------------------------------------------
            # Get Text Fragments
            # ------------------------------------------------
            txt_teeth = list(self.gearbox.Degradation.GearIn_Degradation.state0['tooth'].to_numpy().reshape(-1))
            # # Boxes  of fallen teeth (same style as taken action) - currently unused
            #txt_truth = ' '.join(['%i:' % (i) + u'\u25FB' if i not in dam_teeth else '%i:' % (i) + u'\u25FC' for i in range(int(action_space.n))])
            txt_pred = ' '.join(['%i:' % (i) + u'\u25FB' if i != self.action else '%i:' % (i) + u'\u25FC' for i in range(int(self.action_space.n))])
            lc = '@ %i' % (int(self.gearbox.ga_load_cycle[-1]))
            # ------------------------------------------------
            # Text = fallen teeth + Taken Action + Load cycle
            # ------------------------------------------------
            txt_ansi = 'T: %s | P: %s | %s'% (str(txt_teeth), txt_pred, lc)
            # ------------------------------------------------
            # Add whitespace if a longer text before was printed
            # ------------------------------------------------
            self.render_len_max = max([len(txt_ansi), self.render_len_max])
            len_diff = max([len(txt_ansi) - self.render_len_max, 0])
            txt_ansi += ' ' * len_diff
            print(txt_ansi, end="\r")
        elif mode == 'ansi2': # return terminal-style text representation for running in batch
            # ------------------------------------------------
            # Get Text Fragments
            # ------------------------------------------------
            txt_teeth = list(self.gearbox.Degradation.GearIn_Degradation.state0['tooth'].to_numpy().reshape(-1))
            # # Boxes  of fallen teeth (same style as taken action) - currently unused
            #txt_truth = ' '.join(['%i:' % (i) + u'\u25FB' if i not in dam_teeth else '%i:' % (i) + u'\u25FC' for i in range(int(self.action_space.n))])
            lc = '@ %i' % (int(self.gearbox.ga_load_cycle[-1]))
            # ------------------------------------------------
            # Text = fallen teeth + Taken Action + Load cycle
            # ------------------------------------------------
            txt_ansi = 'Truth: %s | Aktion: %s | Reward: %.3f | %s'% (str(txt_teeth), str(self.action), self.reward, lc)
            # ------------------------------------------------
            # Add whitespace if a longer text before was printed
            # ------------------------------------------------
            self.render_len_max = max([len(txt_ansi), self.render_len_max])
            len_diff = max([len(txt_ansi) - self.render_len_max, 0])
            txt_ansi += ' ' * len_diff
            print(txt_ansi, end="\r")
        else:
            pass
        # ------------------------------------------------
        # If Done true print new line to start new line for next
        # ------------------------------------------------
        if self.done:
            print('\n')

    def close(self):
        pass

# ------------------------------------------------
    def nextseed(self):
        """
        Method to get new seed for next episode,
        different than previous seed.
        """
        seed = dc(self.seed)
        seed += np.random.randint(1, high=10, size=1, dtype=np.int32)[0]
        if seed > 2**16:
            seed = seed - 2**16
        return(seed)


    def check_stop_criteria(self, statei, criteria):
        """
        Method to check if criteria is reached in statei.
        Currently it checks if any gear pitting is >= criteria,
        e.g. 4 %.
        Returns False if stop criteria is not reached
        Returns True if stop criteria is reached
        """
        for key in statei.keys():
            if statei[key] is not None:
                if (statei[key] >= criteria).to_numpy().any():
                    return(False)
        return(True)

    def startpoint_detection(self):
        if self.gearbox.Degradation.GearIn_Degradation.state0 is not None:
            gearin_n0_min = min(self.gearbox.Degradation.GearIn_Degradation.state0['n0'])
        else:
            gearin_n0_min = np.inf
        if self.gearbox.Degradation.GearOut_Degradation.state0 is not None:
            gearout_n0_min = min(self.gearbox.Degradation.GearOut_Degradation.state0['n0'])
        else:
            gearout_n0_min = np.inf

        n0_min = min(gearin_n0_min, gearout_n0_min)
        n0_min = max(n0_min, 0)
        remainder = n0_min % self.nolc_step
        startpoint = n0_min - remainder + self.nolc_step
        return(startpoint)

```

---
# Custom Policy
---

This policy is build to use 'Conv1D' Layers

Import Modules


```python
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.tf_layers import ortho_init
```

The modified Conv1D Layer based on [stable baselines conv definition](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/a2c/utils.py#L37)


```python
def conv1d(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow
    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    # if isinstance(filter_size, list) or isinstance(filter_size, tuple):
    #     assert len(filter_size) == 2, \
    #         "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
    #     filter_height = filter_size[0]
    #     filter_width = filter_size[1]
    # else:
    #     filter_height = filter_size
    #     filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 2
        strides = [1, stride, 1]
        bshape = [1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride]
        bshape = [1, n_filters, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_size, n_input, n_filters]
    with tf.variable_scope(scope):
        # tbd set initilialiser to sensor shaped [0.5, 1, 0.5, 0, 0, 0, 0,] and trainable false
        weight = tf.get_variable("w", wshape, initializer=None)
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv1d(input_tensor, weight, stride=strides, padding=pad, data_format=data_format)
```

Define modified cnn policy [Source](https://github.com/hill-a/stable-baselines/issues/220)

Number of conv layers and for each layer filter_size, stride and number of filters must be defined



```python
def modified_cnn(scaled_images, *args, **kwargs):
    activ = tf.nn.relu
    x = activ(conv1d(scaled_images, 'c1', n_filters=l1_n_filters, filter_size=l1_filter_size, stride=l1_stride, init_scale=np.sqrt(2), **kwargs))
    if no_conv_layer >= 2:
        x = activ(conv1d(x, 'c2', n_filters=l2_n_filters, filter_size=l2_filter_size, stride=l2_stride, init_scale=np.sqrt(2), **kwargs))
    if no_conv_layer >= 3:
        x = activ(conv1d(x, 'c3', n_filters=l3_n_filters, filter_size=l3_filter_size, stride=l3_stride, init_scale=np.sqrt(2), **kwargs))
    if no_conv_layer >= 4:
        x = activ(conv1d(x, 'c4', n_filters=l4_n_filters, filter_size=l4_filter_size, stride=l4_stride, init_scale=np.sqrt(2), **kwargs))
    x = conv_to_fc(x)
    return activ(linear(x, 'fc1', n_hidden=n_hidden, init_scale=np.sqrt(2)))
```

Differ between deepq and common policies

Using 'deepq':


```python
from stable_baselines.deepq.policies import FeedForwardPolicy, CnnPolicy
```

Using 'common':


```python
from stable_baselines.common.policies import FeedForwardPolicy
```

Add ANN to custom Policy


```python
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")
```

---
# Putting Everything together
---


Initialize a new Instance (set verbose to zero):


```python
gearbox = Gearbox(rotational_frequency_in,
                  sample_interval, sample_rate,
                  # Vibration Arguments
                  GearIn, GearOut,
                  Bearing1, Bearing2, Bearing3, Bearing4,
                  # Degradation Arguments
                  Deg_GearIn, Deg_GearOut,
                  Deg_Bearing1, Deg_Bearing2, Deg_Bearing3, Deg_Bearing4,
                  # Shared Arguments
                  seed=seed,
                  verbose=0, # 0: no output of "load cycle #### done"
                  fixed_start=True,
                  GearDegVibDictIn=GearDegVibDictIn,
                  GearDegVibDictOut=GearDegVibDictOut)
```

Make Environment


```python
env = GearboxBaseEnv(gearbox, initial_torque, nolc_step=nolc_step,
                     interpreter=interpreter,
                     vibrations2observations=vibrations2observations,
                     action2torque=action2torque,
                     observation_space=observation_space,
                     action_space=action_space,
                     stop_criteria=4.0, seed=8,
                     render_in_step=True, # using render method in each step
                     warn_limit=30e6, # force env.done if limit is reached
                     verbose=0,# 0:save no infos, 1:save infos each step, 2: save info episode end
                     render_mode='ansi2', #ansi1 more detailes (for ipynb) and ansi2 less detailed (for terminal)
                     # kwargs
                     interpreter_parameters=interpreter_parameters,
                     observation_parameters=observation_parameters, # neccesary, can be empty dict
                     action_parameters=action_parameters) # neccesary, can be empty dict
```

Define Agent

E.g. using 'DQN':


```python
from stable_baselines import DQN
model = DQN(CustomPolicy, env,
            gamma=gamma, learning_rate=learning_rate,
            verbose=1, tensorboard_log=tensorboard_log)
```

E.g. using 'PPO2':


```python
from stable_baselines import PPO2
model = PPO2(CustomPolicy, env,
             gamma=gamma, learning_rate=learning_rate,
             verbose=1, tensorboard_log=tensorboard_log)
```

Learn Model


```python
model.learn(total_timesteps=total_timesteps)
```
