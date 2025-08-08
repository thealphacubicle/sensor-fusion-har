# Sensor Fusion HAR

An experimental project for the DS5220 course using the UCI Human Activity
Recognition (HAR) dataset. The repository trains a suite of classical machine
learning models across different sensor configurations while logging results to
[Weights & Biases](https://wandb.ai/).

## Quick Start

### Installation

1. Create a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the processed training and test data exist under `data/processed/` as
   `train.parquet` and `test.parquet` respectively.

### Configuration

- `config/models.yaml` – enables/disables models and sets default parameters.
- `config/training_config.yaml` – defines notification preferences and
  hyperparameter grids for each model.

### Running Training

Set your Weights & Biases API key in the environment, then execute:

```bash
export WANDB_API_KEY="your-key"
python main.py
```

The script sequentially trains every enabled model on each sensor configuration
and logs metrics such as `train_accuracy` and `test_accuracy` to W&B. Desktop
notifications are sent at the start and end of the experiment and after each
model–sensor pair completes when enabled in the training config.

## Contributing

Issues and pull requests are welcome. Please run `python -m pytest` before
submitting changes.
