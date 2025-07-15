from xgboost import XGBClassifier
from src.models.base import Model


class XGBModel(Model):
    def build_model(self):
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
