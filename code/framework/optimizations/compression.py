# optimizations/compression.py
from sklearn.feature_selection import SelectFromModel

def reduce_features(model, X_train, y_train):
    # Reduces the features of a model by selecting the most important ones
    selector = SelectFromModel(model)
    selector.fit(X_train, y_train)
    X_new = selector.transform(X_train)
    return X_new