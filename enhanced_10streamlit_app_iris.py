
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("Iris Classifier: Decision Tree vs Random Forest")

# Sidebar to choose classifier
classifier = st.sidebar.selectbox("Choose Classifier", ("Decision Tree", "Random Forest"))

# Sidebar hyperparameters
if classifier == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
elif classifier == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 100, 50, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display results
st.subheader("Test Accuracy")
st.write(f"{accuracy_score(y_test, y_pred):.2f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

st.subheader("Feature Input Example")
st.write(X.head())
