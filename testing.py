import Linear_fit
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

a = np.array([[1,4,2],[7,9,4],[0,6,2]])
# np.savetxt('myfile.csv', a, delimiter=',')

df = pd.DataFrame(a)
print(df)
df.to_csv('myfile.csv')

