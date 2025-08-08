from xgboost import XGBClassifier
from src.models.base import Model

class XGBModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def build_model(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **self.kwargs)
