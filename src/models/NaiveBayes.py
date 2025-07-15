from sklearn.naive_bayes import GaussianNB
from src.models.base import Model

class NaiveBayesModel(Model):
    def build_model(self):
        self.model = GaussianNB()
