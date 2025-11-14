from sklearn import datasets
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# load in the iris dataset
iris = datasets.load_iris()

# convert to a pandas dataframe
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# add target
iris_df['target'] = iris.target

# train test split
iris_train_df, iris_test_df = sklearn.model_selection.train_test_split(iris_df, test_size=0.2, random_state = 31779)

# set up the random forest classifier
amodel = RandomForestClassifier(random_state=0)

# create the various datas we need
iris_training_features = iris_train_df[iris.feature_names].copy()
iris_test_features = iris_test_df[iris.feature_names].copy()
iris_target_training = iris_train_df['target'].copy()

# fit the model
amodel.fit(iris_training_features, iris_target_training)

# predict against test
iris_target_testing = iris_test_df['target'].copy()
preds = amodel.predict(iris_test_features)