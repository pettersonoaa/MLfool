# Upper Confidence Bound

import pandas as pd
import matplotlib.pyplot as plt

# import df
df = ...

QTD_ROWS = len(df) # qtd de usuários (rows)
QTD_COLS = len(df.columns) # qtd de items (cols)

item_selected = []
items = [0] * QTD_COLS
sum_rewards = [0] * QTD_COLS
total_reward = 0

for n in range(0, QTD_ROWS):
    
    item = 0
    max_upper_bound = 0
    
    for i in range(0, QTD_COLS):
        
        if items[i] > 0:
            avg_reward = sum_rewards[i] / items[i]
            upper_bound = avg_reward + (math.sqrt(3/2 * math.log(n + 1) / items[i])) # intervalo de confianca superior
        else:
            upper_bound = 1e400 # número muito grande
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            item = i

    item_selected.append(item)
    items[item] += 1
    
    reward = df.values[n, item]
    sum_rewards[item] += reward
    total_reward += reward

plt.hist(item_selected);
plt.xlabel('Items');
plt.ylabel('Freq.');
