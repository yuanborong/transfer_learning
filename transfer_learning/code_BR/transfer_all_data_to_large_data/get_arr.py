import pandas as pd

base = 'Drg'
list = []
for i in range(309 , 311 + 1):
    temp = ''
    temp += (base + str(i))
    list.append(temp)
print(list)

