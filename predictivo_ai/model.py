from sklearn.tree import DecisionTreeClassifier
import joblib


class PredictivoModel:

    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def feature_importances(self):
        return self.model.feature_importances_

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
