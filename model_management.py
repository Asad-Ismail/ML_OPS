import logging
import requests
import zipfile
import io
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from pydantic import BaseModel, ValidationError, field_validator
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PipelineInput(BaseModel):
    """
    Represents the input required to configure a machine learning pipeline.
    
    Attributes:
        pipeline_id (str): Identifier for the pipeline.
        hyperparameters (dict): Hyperparameters for the pipeline's model.
    """
    pipeline_id: str
    hyperparameters: dict

    @field_validator('hyperparameters')
    def check_lr(cls, v):
        """Validates that the learning rate is greater than zero."""
        if ('lr' in v and v['lr'] <= 0) or ('learning_rate' in v and v['learning_rate'] <= 0):
            raise ValueError('Learning rate (lr) must be greater than 0')
        return v
    
class InferenceInput(BaseModel):
    """
    Represents the input required for model inference.

    Attributes:
        pipeline_id (str): Identifier for the pipeline to use for inference.
        new_data (list): New data to be used for inference.
    """
    pipeline_id: str
    new_data: list

class TrainInput(BaseModel):
    """
    Represents the training data input for a machine learning model.

    Attributes:
        X_data (list): The feature data for training.
        Y_data (list): The target data for training.
    """
    X_data: list
    Y_data: list

class ValidInput(BaseModel):
    """
    Represents the validation data input for evaluating a machine learning model.

    Attributes:
        pipeline_id (str): Identifier for the pipeline to be evaluated.
        X_data (list): The feature data for validation.
        Y_data (list): The target data for validation.
    """
    pipeline_id: str
    X_data: list
    Y_data: list

def get_data():
    """
    Downloads and extracts the dataset from a specified URL.
    """
    url = "https://s3-us-west-2.amazonaws.com/sagemaker-e2e-solutions/fraud-detection/creditcardfraud.zip"
    try:
        logging.info("Starting to download the file...")
        response = requests.get(url)
        zip_content = response.content
        logging.info("Download complete. Extracting files...")

        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            zip_ref.extractall("creditcardfraud")
        logging.info("Successfully extracted files")
    except requests.RequestException as e:
        logging.error(f"Error during downloading: {e}")
    except zipfile.BadZipFile as e:
        logging.error(f"Error during extraction: {e}")

def load_data():
    """
    Loads the dataset, either by downloading it or reading a local file.
    Splits the data into training and validation sets.

    Returns:
        tuple: A tuple containing the training and validation data.
    """
    if not os.path.exists('creditcardfraud/creditcard.csv'):
        get_data()

    model_data = pd.read_csv('creditcardfraud/creditcard.csv', delimiter=',')
    model_data.reset_index(inplace=True)

    X = model_data.drop('Class', axis=1).values
    y = model_data['Class'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    logging.info(f"Data loaded with training shape: {X_train.shape}")

    return X_train, X_val, y_train, y_val


class ModelManager:
    """
    Manages machine learning pipelines including training, evaluation, and inference.

    Attributes:
        fitted_pipelines (dict): Stores fitted pipeline models.
        data_properties (dict): Stores properties of training data for each pipeline.
        model_metrics (dict): Stores evaluation metrics for each pipeline.
    """

    def __init__(self):
        self.fitted_pipelines = {}
        self.data_properties = {}
        self.model_metrics = {}

    def pipeline_fit(self, pipeline_input: PipelineInput, data: TrainInput):
        """
        Trains and stores a machine learning pipeline based on provided input.

        Args:
            pipeline_input (PipelineInput): Configuration for the pipeline.
            data (TrainInput): Training data.
        """
        try:
            input_data = PipelineInput(**pipeline_input)
            train_data = TrainInput(**data)

            if input_data.pipeline_id.lower() == 'xgboost':
                estimator = XGBClassifier(**input_data.hyperparameters)
                pipeline = Pipeline(steps=[('xgboost_classifier', estimator)])
            else:
                raise ValueError(f"Unsupported pipeline ID: {input_data.pipeline_id}")

            X_data_np = np.array(train_data.X_data)
            Y_data_np = np.array(train_data.Y_data)
            fitted_model = pipeline.fit(X_data_np, Y_data_np)
            logging.info("Model Trained!!")

            self.fitted_pipelines[input_data.pipeline_id] = fitted_model
            self.data_properties[input_data.pipeline_id] = self._analyze_data(X_data_np)

        except ValidationError as e:
            logging.error(f"Error in pipeline input: {e}")

    def pipeline_evaluate(self, validation_input: ValidInput):
        """
        Evaluates the pipeline on validation data and stores the metrics.

        Args:
            validation_input (ValidInput): Validation data along with pipeline identifier.
        """
        try:
            valid_data = ValidInput(**validation_input)
            pipeline_id = valid_data.pipeline_id

            if pipeline_id not in self.fitted_pipelines:
                raise ValueError("Pipeline not found")

            model = self.fitted_pipelines[pipeline_id]
            predictions = model.predict(np.array(valid_data.X_data))

            precision = precision_score(valid_data.Y_data, predictions)
            recall = recall_score(valid_data.Y_data, predictions)
            f1 = f1_score(valid_data.Y_data, predictions)

            logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            self.model_metrics[pipeline_id] = {
                'evaluation_metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            }

        except ValidationError as e:
            logging.error(f"Error in validation input: {e}")

    def inference(self, inference_input: InferenceInput):
        """
        Performs inference using a specified pipeline.

        Args:
            inference_input (InferenceInput): Inference data along with pipeline identifier.

        Returns:
            list: Predictions made by the model.
        """
        try:
            

            input_data = InferenceInput(**inference_input)
            # Analyze Input

            self._monitor_data(np.array(input_data.new_data), input_data.pipeline_id)

            # Perform Inference
            if input_data.pipeline_id not in self.fitted_pipelines:
                raise ValueError("Pipeline not found")

            model = self.fitted_pipelines[input_data.pipeline_id]
            predictions = model.predict(np.array(input_data.new_data))
            return predictions

        except ValidationError as e:
            logging.error(f"Error in inference input: {e}")

    def _analyze_data(self, data):
        """
        Analyzes and stores statistical properties of the data.

        Args:
            data (np.ndarray): Data to be analyzed.

        Returns:
            dict: Properties of the data (mean, standard deviation).
        """
        return {'mean': np.mean(data, axis=0), 'std': np.std(data, axis=0)}

    def _monitor_data(self, new_data, pipeline_id):
        """
        Monitors the new data by comparing its properties to the stored data properties.

        Args:
            new_data (list): New data for comparison.
            pipeline_id (str): Identifier for the pipeline whose data properties to compare against.
        """
        train_properties = self.data_properties[pipeline_id]
        infer_properties = self._analyze_data(np.array(new_data))

        comapre_std=4
        # checking if mean and std is with in thresold std away as more than 95 % of data points should within  this
        mean_upper_threshold = train_properties['mean'] + comapre_std * train_properties['std']
        mean_lower_threshold = train_properties['mean'] - comapre_std * train_properties['std']
        std_upper_threshold = train_properties['std'] + comapre_std * np.std(train_properties['std'])
        std_lower_threshold = train_properties['std'] - comapre_std * np.std(train_properties['std'])

        if np.any(infer_properties['mean'] > mean_upper_threshold) or np.any(infer_properties['mean'] < mean_lower_threshold):
            logging.warning("Significant deviation in mean detected")

        if np.any(infer_properties['std'] > std_upper_threshold) or np.any(infer_properties['std'] < std_lower_threshold):
            logging.warning("Significant deviation in standard deviation detected")


if __name__ == "__main__":
    
    X_train, X_val, y_train, y_val = load_data()

    # Prepare training input
    train_input = {
        'X_data': X_train.tolist(),
        'Y_data': y_train.tolist()
    }

    # Pipeline configuration
    pipeline_input = {
        'pipeline_id': 'xgboost',
        'hyperparameters': { 
            'max_depth': 3, 
            'n_estimators': 100, 
            'learning_rate': 0.1, 
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
    }

    # Initialize model manager
    mm = ModelManager()

    # Train model
    logging.info("Training Model")
    mm.pipeline_fit(pipeline_input, train_input)

    # Validate model
    valid_input = {
        'pipeline_id': 'xgboost',
        'X_data': X_val.tolist(),
        'Y_data': y_val.tolist()
    }
    mm.pipeline_evaluate(valid_input)

    # Perform inference on validation data
    logging.info("Performing Inference on Validation Data")
    predictions = []
    for x_val in tqdm(X_val, desc="Inferencing", unit="sample"):
        inference_input = {'pipeline_id': 'xgboost', 'new_data': [x_val.tolist()]}
        prediction = mm.inference(inference_input)
        predictions.append(prediction)
