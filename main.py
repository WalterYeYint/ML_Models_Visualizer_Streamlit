import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Streamlit Hey There")

st.write("""
# Explore different classifier
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))



def get_dataset(dataset_name):
	if dataset_name == "Iris":
		data = datasets.load_iris()
	elif dataset_name == "Breast Cancer":
		data = datasets. load_breast_cancer()
	else:
		data = datasets.load_wine()
	x = data.data
	y = data.target
	return x, y

x, y = get_dataset(dataset_name)
st.write("shape of dataset", x.shape)
st.write("number of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
	params = dict()
	if clf_name == "KNN":
		K = st.sidebar.slider("K", 1, 15)
		params["K"] = K
	elif clf_name == "SVM":
		C = st.sidebar.slider("C", 0.01, 10.0)
		params["C"] = C
	else:
		max_depth = st.sidebar.slider("max depth", 2, 15)
		n_estimators = st.sidebar.slider("number of estimators", 1, 100)
		params["max_depth"] = max_depth
		params["n_estimators"] = n_estimators
	return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
	if clf_name == "KNN":
		clf = KNeighborsClassifier(n_neighbors=params["K"])
	elif clf_name == "SVM":
		clf = SVC(C=params["C"])
	else:
		clf = RandomForestClassifier(n_estimators=params["n_estimators"],
																max_depth=params["max_depth"], random_state=1234)
	return clf

clf = get_classifier(classifier_name, params)

# Classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)

# This print doesn't work if "f" is not involved
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")


