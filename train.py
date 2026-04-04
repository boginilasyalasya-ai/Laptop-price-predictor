import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


data = {
    'RAM': [4, 8, 16, 32],
    'Storage': [256, 512, 512, 1024],
    'SSD': [0, 1, 1, 1],
    'Price': [30000, 50000, 75000, 120000]
}

df = pd.DataFrame(data)

X = df[['RAM', 'Storage', 'SSD']]
y = df['Price']


model = LinearRegression()
model.fit(X, y)


pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully!")
