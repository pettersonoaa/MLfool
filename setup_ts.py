# Thompson Sampling

import pandas as pd
import matplotlib.pyplot as plt
import random

# import df
df = ...

QTD_ROWS = len(df) # qtd de usuários (rows)
QTD_COLS = len(df.columns) # qtd de items (cols)

item_selected = []
qtd_reward_0 = [0] * QTD_COLS
qtd_reward_1 = [0] * QTD_COLS
total_reward = 0

for n in range(0, QTD_ROWS):
    
    item = 0
    max_random = 0
    
    for i in range(0, QTD_COLS):
        
        random_beta = random.betavariate(qtd_reward_1[i] + 1, qtd_reward_0[i] + 1)
        
        if random_beta > max_random:
            max_random = random_beta
            item = i

    item_selected.append(item)
    reward = df.values[n, item]
    total_reward += reward
    if reward == 1:
        qtd_reward_1[item] += 1
    else:
        qtd_reward_0[item] += 1
        
plt.hist(item_selected);
