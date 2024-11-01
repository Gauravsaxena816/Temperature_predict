import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('weather_data.csv')

X = data[['Relative Humidity', 'Pressure']]
y = data['Temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model accuracy: {score}")


# Save the model to a file
with open('temperature_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as 'temperature_model.pkl'")
