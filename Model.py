from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -------- NEW DATA --------
# New flower measurements (not from dataset)
new_data = [
    [5.0, 3.4, 1.5, 0.2],   # sample 1
    [6.5, 3.0, 5.2, 2.0],   # sample 2
    [5.8, 2.7, 4.1, 1.0]    # sample 3
]

# Predict
predictions = rf.predict(new_data)
probabilities = rf.predict_proba(new_data)

# Output
for i, p in enumerate(predictions):
    print(f"Sample {i+1}")
    print(" Predicted class:", iris.target_names[p])
    print(" Probabilities:", probabilities[i])