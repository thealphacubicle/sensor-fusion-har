# Sensor Fusion HAR

An experimental project for the DS5220 course using the UCI Human Activity Recognition (HAR) dataset. The repository trains a suite of classical machine learning models across different sensor configurations while logging results to [Weights & Biases](https://wandb.ai/).

## Quick Start

### Installation

1. Create a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the processed training and test data exist under `data/processed/` as `train.parquet` and `test.parquet` respectively.

### Configuration

* `config/models.yaml` – enables/disables models and sets default parameters.
* `config/training_config.yaml` – defines notification preferences and hyperparameter grids for each model.

### Running Training

1. Create a `.env` file in the root of the repository with your Weights & Biases API key:

   ```
   WANDB_API_KEY=your-key
   ```

2. Run the data seed script to load the dataset locally:

   ```bash
   python src/utils/preprocess.py
   ```

3. Start the experiment:

   ```bash
   python main.py
   ```

The script sequentially trains every enabled model on each sensor configuration and logs metrics such as `train_accuracy` and `test_accuracy` to W&B. Desktop notifications are sent at the start and end of the experiment and after each model–sensor pair completes, when enabled in the training config.

### Dashboard

A lightweight dashboard built with [Reflex](https://reflex.dev) visualizes training and test accuracy from the W&B project.

1. Ensure `WANDB_API_KEY` (and optionally `WANDB_ENTITY`) is set in your environment.
2. Run the dashboard:

   ```bash
   reflex run src/dashboard/app.py
   ```

The page will display line charts of training and test accuracy for each run logged to W&B.

## Contributing

Issues and pull requests are welcome!

## Feedback and Questions
For any questions or feedback, please open an issue or contact [me via email!](mailto:raman.sr@northeastern.edu)
