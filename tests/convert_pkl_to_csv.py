import pickle as pkl
import pandas as pd

with open("C:\\Users\\ruyas\\code\\fetModeling\\data\\augmented\\augmented_data.pkl", "rb") as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
df.to_csv(r'C:\\Users\\ruyas\\code\\fetModeling\\data\\augmented\\augmented_data.csv')