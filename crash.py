import pandas as pd
import numpy as np

np.random.seed(42)
n=1000
ages = np.random.randint(16,90, size=n)
speeds = np.random.randint(10,120, size=n)

prob_survival = 1/(1+np.exp((ages-30)/15+(speeds-30)/20))
survived = np.random.binomial(1,prob_survival)

df = pd.DataFrame(
    {
        'Age':ages,
        'Speed':speeds,
        'Survived':survived
    })
df.to_csv('crash.csv',index=False)
print("Simulated crash.csv has been created")
