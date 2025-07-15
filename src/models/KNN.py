from sklearn.neighbors import KNeighborsClassifier
from src.models.base import Model


class KNNModel(Model):
    def build_model(self):
        self.model = KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            metric='minkowski'  # Default Euclidean
        )
