
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib
import time
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
import numpy as np


preprocessor = joblib.load("data/preprocessor.joblib")

target = ["gnd Genre"]


# debugging
start_time = time.time()
# get features and target
# 0 to NaN for median parsing on numerical data

# init random forest model
SVM = LinearSVC(
    random_state=67,
    dual=False
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVM,)
])
# make parameter grid
param_dist = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__max_iter": [1000, 3000, 5000]
}
# grid search, look for best weighted f1 score
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=3,
    cv=3,
    scoring="f1_weighted",
    #n_jobs=-1,
    verbose=2,
    error_score='raise',
    random_state=67
)
# fit
df = pd.read_csv('data/converted_data/b3_final_ger', sep=';')
subset = df.groupby('gnd Genre', group_keys=False).sample(frac=0.1, random_state=67)

X = subset[["Publisher", "Title Statement", "Author", "gnd Topical Term", "Date", "Number of Pages"]].fillna("")
y = subset['gnd Genre']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=67,
    stratify=y
)
random_search.fit(X_train, y_train)
# best results
print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)
# best model
best_model = random_search.best_estimator_
print("--- %s seconds ---" % (time.time() - start_time))
# save best model

joblib.dump(best_model, "SVM.pkl")


