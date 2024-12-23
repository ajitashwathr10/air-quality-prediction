import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class ChaoticSystemAnalyzer:
    def __init__(self, seed: int = 42):
        """Initialize the analyzer with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.scaler = MinMaxScaler()
        
    def generate_lorenz_attractor(self, n_points: int = 1000) -> np.ndarray:
        """
        Generate Lorenz attractor time series data.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array containing the Lorenz attractor coordinates
        """
        dt = 0.01
        sigma, rho, beta = 10, 28, 8/3
        
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.zeros(n_points)
        
        x[0], y[0], z[0] = np.random.rand(3)
        
        for i in range(1, n_points):
            dx = sigma * (y[i-1] - x[i-1])
            dy = x[i-1] * (rho - z[i-1]) - y[i-1]
            dz = x[i-1] * y[i-1] - beta * z[i-1]
            
            x[i] = x[i-1] + dx * dt
            y[i] = y[i-1] + dy * dt
            z[i] = z[i-1] + dz * dt
            
        return np.column_stack((x, y, z))
    
    def calculate_lyapunov_exponent(self, time_series: np.ndarray, 
                                  embedding_dim: int = 3, 
                                  delay: int = 1) -> float:
        """
        Calculate the largest Lyapunov exponent of a time series.
        
        Args:
            time_series: Input time series data
            embedding_dim: Embedding dimension
            delay: Time delay
            
        Returns:
            Estimated largest Lyapunov exponent
        """
        n = len(time_series)
        embedded = np.zeros((n - (embedding_dim-1)*delay, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = time_series[i*delay:n-(embedding_dim-1-i)*delay]
            
        distances = np.zeros((len(embedded), len(embedded)))
        for i in range(len(embedded)):
            for j in range(len(embedded)):
                distances[i,j] = np.linalg.norm(embedded[i] - embedded[j])
                
        np.fill_diagonal(distances, np.inf)
        nearest_neighbors = np.argmin(distances, axis=1)
        
        divergence_rates = []
        for i in range(len(nearest_neighbors)-1):
            div = np.log(abs(time_series[i+1] - time_series[nearest_neighbors[i]+1]))
            if not np.isinf(div) and not np.isnan(div):
                divergence_rates.append(div)
                
        return np.mean(divergence_rates)
    
    def build_neural_network(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build a hybrid CNN-LSTM neural network model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled neural network model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation = 'relu',
                                 input_shape = input_shape),
            tf.keras.layers.MaxPooling1D(pool_size = 2),
            tf.keras.layers.LSTM(100, return_sequences = True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(32, activation = 'relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
        return model
    
    def prepare_sequences(self, data: np.ndarray, 
                         lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series prediction.
        
        Args:
            data: Input time series data
            lookback: Number of previous time steps to use
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i : i+lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
    
    def calculate_complexity_metrics(self, time_series: np.ndarray) -> Dict:
        """
        Calculate various complexity metrics for the time series.
        
        Args:
            time_series: Input time series data
            
        Returns:
            Dictionary containing complexity metrics
        """
        def sample_entropy(ts, m=2, r=0.2):
            n = len(ts)
            templates = np.array([ts[i : i+m] for i in range(n - m + 1)])
            distances = np.zeros((len(templates), len(templates)))
            
            for i in range(len(templates)):
                for j in range(len(templates)):
                    distances[i,j] = np.max(np.abs(templates[i] - templates[j]))
                    
            matches_m = np.sum(distances < r, axis=1)
            matches_m1 = np.sum(distances[:, :-1] < r, axis=1)[:-1]
            
            return -np.log(np.sum(matches_m1) / np.sum(matches_m))

        metrics = {
            'lyapunov': self.calculate_lyapunov_exponent(time_series),
            'sample_entropy': sample_entropy(time_series),
            'kurtosis': pd.Series(time_series).kurtosis(),
            'histogram_entropy': entropy(np.histogram(time_series, bins=50)[0])
        }
        return metrics
    
    def analyze_and_predict(self, save_plots: bool = True) -> Dict:
        """
        Perform complete analysis including data generation, complexity analysis,
        and prediction.
        
        Args:
            save_plots: Whether to save visualization plots
            
        Returns:
            Dictionary containing analysis results
        """
        data = self.generate_lorenz_attractor()
        
        scaled_data = self.scaler.fit_transform(data)
        metrics = self.calculate_complexity_metrics(scaled_data[:, 0])
        X, y = self.prepare_sequences(scaled_data[:, 0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        model = self.build_neural_network((X.shape[1], 1))
        history = model.fit(X_train, y_train, epochs = 50, batch_size = 32,
                          validation_split = 0.2, verbose = 0)
        
        predictions = model.predict(X_test)

        if save_plots:
            self._create_visualizations(data, scaled_data, history, 
                                     y_test, predictions)
        
        return {
            'complexity_metrics': metrics,
            'model_history': history.history,
            'predictions': predictions,
            'actual_values': y_test
        }
    
    def _create_visualizations(self, data: np.ndarray, 
                             scaled_data: np.ndarray,
                             history: tf.keras.callbacks.History,
                             y_test: np.ndarray,
                             predictions: np.ndarray) -> None:
        """Create and save visualization plots."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:, 0], data[:, 1], data[:, 2])
        ax.set_title('Lorenz Attractor')
        plt.savefig('lorenz_attractor.png')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()

        plt.figure(figsize = (10, 6))
        plt.plot(y_test, label = 'Actual')
        plt.plot(predictions, label = 'Predicted')
        plt.title('Predictions vs Actual Values')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('predictions.png')
        plt.close()

if __name__ == "__main__":
    analyzer = ChaoticSystemAnalyzer()
    results = analyzer.analyze_and_predict()
    
    print("\nComplexity Metrics:")
    for metric, value in results['complexity_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nFinal Training Loss:", results['model_history']['loss'][-1])
    print("Final Validation Loss:", results['model_history']['val_loss'][-1])