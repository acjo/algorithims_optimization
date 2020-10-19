#osman_caelan_4_5.py

from itertools import combinations


def knapsack(W, items):
    indicies = []
    possibilities = []
    for i in range(1, len(items)+1):
        combination = list(combinations(items, i))
        for comb in combination:
            combination_weight = 0
            for sub in comb:
                combination_weight += sub[0]
            if combination_weight <= W:
                possibilities.append(comb)
    max_value = 0
    for possible in possibilities:
        value = 0
        for sub in possible:
            value += sub[1]
        if value > max_value:
            max_value = value
            final_combination = possible
    for item in final_combination:
        x = items.index(item)
        indicies.append(x)
    return max_value, indicies



print(knapsack(120, [(20, 0.5), (50, 1), (100, 1.2)]))
