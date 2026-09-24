import pickle
import numpy as np

m = pickle.load(open("model.pkl", "rb"))
s = pickle.load(open("scaler.pkl", "rb"))
row = np.array([[2000, 3, 2, 2005, 2.5, 1, 7]])
price = float(m.predict(s.transform(row))[0])
print("Predicted price: USD", round(price, 2))
print("Flask API is ready to serve predictions.")
