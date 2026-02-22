# import pandas as pd
# from aptrade.domain.strategy.sma_crossover import SMACrossStrategy

# prices = pd.Series([
#     100,101,102,103,104,
#     105,106,107,108,109,
#     108,107,106,105,104,
#     103,102,101,100,99,
#     98,97,96,95,94,
#     93,92,91,90,89,
#     88,87,86,85,84
# ])

# strategy = SMACrossStrategy(fast=3, slow=5)

# signals = []
# has_position = False

# for price in prices:
#     decision = strategy.on_event(price, has_position)
#     if decision:
#         signals.append(decision.action)
#         if decision.action == "buy":
#             has_position = True
#         elif decision.action == "close":
#             has_position = False

# print("Signals:", signals)
