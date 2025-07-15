from sklearn.linear_model import LogisticRegression
from src.models.base import Model

class LogisticRegressionModel(Model):
    def build_model(self):
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
