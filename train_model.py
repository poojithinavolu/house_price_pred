import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create sample house data
data = {
    "area": [
        500, 600, 750, 800, 900,
        1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 2000,
        2200, 2400, 2600, 2800, 3000
    ],
    "bedrooms": [
        1, 1, 2, 2, 2,
        2, 3, 3, 3, 3,
        3, 3, 4, 4, 4,
        4, 4, 5, 5, 5
    ],
    "age": [
        15, 12, 10, 8, 10,
        7, 8, 6, 5, 7,
        5, 4, 6, 3, 4,
        3, 2, 3, 2, 1
    ],
    # Sample prices in Indian rupees
    # These are illustrative, not actual market prices.
    "price": [
        2000000, 2400000, 3000000, 3200000, 3500000,
        4000000, 4200000, 4700000, 5000000, 5200000,
        5700000, 6100000, 6500000, 7000000, 7600000,
        8200000, 9000000, 9600000, 10500000, 11500000
    ]
}

df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# 3. Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model on unseen test data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model training completed!")
print(f"Mean Absolute Error: ₹{mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# 6. Save the trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as house_price_model.pkl")
