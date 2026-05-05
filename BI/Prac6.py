import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Decision_Tree_Dataset.csv")

# Features & target
X = df.drop("Result", axis=1)
y = df["Result"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier(criterion="gini", max_depth=4)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# ---- Visualize Tree ----
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=X.columns)
plt.show()