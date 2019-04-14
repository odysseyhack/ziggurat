
import numpy as np
from random import random 
import pandas as pd
import math

# Lists
team =[]
network = []
hackathon = []
innovation_hub = []
market_fit = []
businessmodel = []
success = []

# semi-random function for giving 1-10
def data_generator(num, list1 = []):
    for i in range(50):
        a = random()
        if a < num: 
            list1.append(math.floor((1-a)*10))
        else:
            list1.append(math.floor(a*10))
    return list1

# semi-random function for giving 0-1
def data_generator_0_1(num, list1 = []):
    for i in range(50):
        a = random()
        if a < num: 
            list1.append(0)
        else:
            list1.append(1)
    return list1

# Make features
data_generator(0.6, team)
data_generator(0.4, network)
data_generator_0_1(0.4, hackathon)
data_generator_0_1(0.3, innovation_hub)
data_generator(0.5, market_fit)
data_generator(0.5, businessmodel)

#Final dataset
dataset = pd.DataFrame({'Team': team,
                      'Network': network,
                      'Hackathon': hackathon,
                      'Innovation Hub': innovation_hub,
                      'Market Fit': market_fit,
                      'Business model' : businessmodel,
                      'Success': success}                       
                      )

