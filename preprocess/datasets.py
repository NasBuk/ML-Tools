import pandas as pd
import numpy as np
import tensorflow as tf
from math import isclose
from datastructures import EasyDict
from preprocess.prepare import normalize_columns
from typing import List, Dict, Tuple


class TFTimeSeriesWindow:
    def __init__(self, data: pd.DataFrame, input_width: int, label_width: int, batch_size: int, label_columns: List[str] = None):
        """
        Initialize the time series window object.

        Args:
            data (pd.DataFrame): The input data frame containing all series.
            input_width (int): The width of the input time window.
            label_width (int): The width of the label time window.
            batch_size (int): The size of batches for the dataset.
            label_columns (list of str): Optional, columns to be used as labels.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if not isinstance(input_width, int) or input_width <= 0:
            raise ValueError("Input width must be a positive integer")
        if not isinstance(label_width, int) or label_width <= 0:
            raise ValueError("Label width must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        if label_columns:
            if not all(isinstance(col, str) for col in label_columns):
                raise ValueError("Label columns must be a list of strings")
            missing_columns = [col for col in label_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Label columns not found in data: {missing_columns}")
            
        self.label_columns = label_columns
        self.label_columns_indices = self._get_column_indices(label_columns) if label_columns else {}
        self.column_indices = {name: i for i, name in enumerate(data.columns)}
        self._setup_window_properties(input_width, label_width)
        
        # Preparing datasets
        try:
            dataset_dictionary = create_split_dataset_dict(data, sequence_length=self.total_window_size, batch_size=batch_size, data_type='timeseries')
            for name, dataset in dataset_dictionary.items():
                setattr(self, name, self.make_dataset(dataset))
        except Exception as e:
            raise RuntimeError(f"Failed to create dataset: {str(e)}")

    def _get_column_indices(self, columns: List[str]) -> Dict[str, int]:
        # Dictionary mapping column names to their indices.
        return {name: i for i, name in enumerate(columns)}

    def _setup_window_properties(self, input_width: int, label_width: int):
        # Set up properties for windowing the data.
        self.input_width = input_width
        self.label_width = label_width
        self.total_window_size = input_width + label_width
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self) -> str:
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split features into inputs and labels as per the defined slices.
        
        Args:
            features (tf.Tensor): Input tensor to be split.
            
        Returns:
            tuple: Tuple of (inputs, labels) tensors.
        """
        try:
            inputs = features[:, self.input_slice, :]
            labels = features[:, self.labels_slice, :]
            if self.label_columns:
                labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])
            return inputs, labels
        except Exception as e:
            raise RuntimeError(f"Failed split window method: {str(e)}")

    def make_dataset(self, data: tf.data.Dataset) -> tf.data.Dataset:
        # Map the split_window method over the dataset.
        return data.map(self.split_window)

    def make_predictions(self, model: tf.keras.Model) -> np.array:
        """
        Use the trained model to make predictions on the test set.
        
        Args:
            model (tf.keras.Model): The trained TensorFlow model.
            
        Returns:
            np.array: Predictions made by the model.
        """
        try:
            predictions = []
            for batch in getattr(self, 'test', []):  # Assuming 'self.test' is your test dataset
                inputs, _ = batch
                batch_predictions = model.predict(inputs)
                predictions.append(batch_predictions)
            
            # Flatten the list of predictions and return as a numpy array
            return np.concatenate(predictions, axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to make predictions: {str(e)}")


def create_split_dataset_dict(data, sequence_length, batch_size, data_type, 
                               split_ratio=[0.7, 0.2, 0.1], shuffle=True, framework="tensorflow"):
    """
    Splits the dataframe into train, validation, and test sets and converts them 
    into TensorFlow datasets based on the specified data type.

    Args:
        data (pd.DataFrame): The input DataFrame containing data.
        sequence_length (int): The length of the output sequences (in number of timesteps).
        batch_size (int): Number of samples in each batch.
        data_type (str): The type of data ('timeseries', 'nlp', 'image').
        split_ratio (List[float], optional): Ratio to split data into train, validation, and test sets.
        shuffle (bool, optional): Whether to shuffle the training dataset.
        framework (str, optional): The ML framework to use ('tensorflow', 'pytorch').

    Returns:
        An EasyDict object containing train, validation, and test TensorFlow datasets.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    if not isinstance(split_ratio, list) or len(split_ratio) != 3:
        raise ValueError("Split ratio must be a list of three numbers.")

    if not isclose(sum(split_ratio), 1.0, rel_tol=1e-9):
        raise ValueError("Split ratios must sum to 1, within a small tolerance.")

    dataset_length = len(data)
    split1 = int(split_ratio[0] * dataset_length)
    split2 = int((split_ratio[0] + split_ratio[1]) * dataset_length)

    datasets = EasyDict(train = data[:split1],
                        val = data[split1:split2],
                        test = data[split2:])

    # Apply the normalization function to each subset in the dictionary
    for subset_name, subset_data in datasets.items():
        datasets[subset_name] = normalize_columns(subset_data, subset_data.columns)

    match framework:
      case 'tensorflow':
        match data_type:
            case 'timeseries' | 'nlp':
                for name, dataset in datasets.items():
                    dataset_np = np.array(dataset, dtype=np.float32)
                    datasets[name] = tf.keras.utils.timeseries_dataset_from_array(
                        data=dataset_np, targets=None, sequence_length=sequence_length,
                        sequence_stride=1, shuffle=shuffle if name == "train" else False,
                        batch_size=batch_size)
            case 'image':
                # Placeholder for image data processing and dataset creation
                pass
            case _:
                raise ValueError(f"Unsupported data type: {data_type}")
      case 'pytorch':
          # PyTorch specific dataset creation logic (placeholder)
          pass
      case _:
          raise ValueError(f"Unsupported framework: {framework}")

    return datasets
