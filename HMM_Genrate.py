#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import decimal
from decimal import Decimal
import math
import time
'''importing alpha beta currency for ts estimation
'''

class MarkovChain(object):
    def __init__(self, transition_prob, emission_prob):
        """
        Initialize the MarkovChain instance.
 
        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition 
            probabilities in Markov Chain. 
            Should be of the form: 
                {'state1': {'state1': 0.1, 'state2': 0.4}, 
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.states = list(transition_prob.keys())
        self.emitted_states = list(emission_prob.keys())
        print(self.emitted_states)
 
    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.
 
        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
            self.states, 
            p=[self.transition_prob[current_state][next_state] 
               for next_state in self.states]
        )
 
    def next_emitted_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.
 
        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        #print("current_state   ",current_state)
        return np.random.choice(
            self.emitted_states, 
            p=[self.emission_prob[emitted_state][current_state] 
               for emitted_state in self.emitted_states]
        )
        
    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.
 
        Parameters
        ----------
        current_state: str
            The state of the current random variable.
 
        no: int
            The number of future states to generate.
        """
        future_states = []
        emitted_states=[]
        x=[]
        for i in range(no):
            next_state = self.next_state(current_state)
            #print("Next state is",next_state)
            emitted_states.append(self.next_emitted_state(next_state))
            future_states.append(next_state)
            current_state = next_state
        x=emitted_states
        return [x[-1], current_state]

def Partition(value):
    start_value=0
    end_value = pow(2,16)-1  # maximam value
    if(value >= 0 and value <end_value/10):
        return 1;
    elif(value >= end_value/10 and value < 2*(end_value/10)):
        return 2;
    elif(value >= 2*(end_value/10) and value < 3*(end_value/10)):
        return 3;
    elif(value >= 3*(end_value/10) and value < 4*(end_value/10)):
        return 4;
    elif(value >= 4*(end_value/10) and value < 5*(end_value/10)):
        return 5;
    elif(value >= 5*(end_value/10) and value < 6*(end_value/10)):
        return 6;
    elif(value >= 6*(end_value/10) and value < 7*(end_value/10)):
        return 7;
    elif(value >= 7*(end_value/10) and value < 8*(end_value/10)):
        return 8;
    elif(value >= 8*(end_value/10) and value < 9*(end_value/10)):
         return 9;
    elif(value >= 9*(end_value/10) and value < (end_value)):
        return 10;
    else:
        return -1;

transition_prob = {'H': {'H': 0.9, 'NH': 0.1}, 'NH': {'H': 0.1, 'NH': 0.9}}

B = {'1': {'H': 0.05, 'NH': 0.2},
     '2': {'H': 0.05, 'NH': 0.2},
     '58932': {'H': 0.05, 'NH': 0.1},
     '4': {'H': 0.05, 'NH': 0.1},
     '5': {'H': 0.1, 'NH': 0.1},
     '30494': {'H': 0.1, 'NH': 0.1},
     '7': {'H': 0.1, 'NH': 0.05},
     '8': {'H': 0.1, 'NH': 0.05},
     '9': {'H': 0.2, 'NH': 0.05},
     '16060': {'H': 0.2, 'NH': 0.05}
     }


chain = MarkovChain(transition_prob=transition_prob,emission_prob=B)
[All_Samples, cur_state]=chain.generate_states(current_state='H', no=1)
#print(All_Samples)

timeout=2.0
initial_time = time.time()

current_time=0.0
slept_time=0

sequence=dict()
vedba=dict()

for index in range(100):
    slept_time=int(np.random.exponential(timeout)+1)
    [All_Samples, cur_state]=chain.generate_states(current_state=cur_state, no=1)
    current_time = current_time + slept_time
    #print(cur_state,All_Samples)
    print(Partition(int(All_Samples)))


