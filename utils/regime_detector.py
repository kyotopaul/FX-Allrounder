import numpy as np
import pandas as pd
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

class RegimeDetector:
    """
    Detects market regimes (e.g., Trending vs Ranging) using a Hidden Markov Model.
    Trains on the combined log returns of multiple USD pairs to find a global USD regime.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.current_regime = 0
        self.current_probability = 0.5
        self.state_probabilities = None

    def fit(self, df):
        """
        Fits HMM on log returns.
        df should contain only the log_ret columns of the pairs.
        """
        if GaussianHMM is None:
            return False

        # Drop NaNs and scale for numerical stability
        data = df.dropna().values * 10000.0

        if len(data) < 100:
            return False

        # Create a fresh model on each scheduled retraining
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=200,
            min_covar=1e-6,
            random_state=42
        )

        # Train the model
        self.model.fit(data)

        states = self.model.predict(data)

        unique, counts = np.unique(states, return_counts=True)

        print("\n===== HMM STATE DISTRIBUTION =====")
        for s, c in zip(unique, counts):
            print(f"State {s}: {c} samples ({100*c/len(states):.1f}%)")
        print("==================================\n")

        return True

    def predict(self, df):
        """
        Predict current hidden state and confidence.
        """

        if self.model is None or GaussianHMM is None:
            return 0

        data = df.dropna().values * 10000.0

        if len(data) == 0:
            return 0

        # Hidden state
        state = self.model.predict(data)[-1]

        # Probability of each state
        probabilities = self.model.predict_proba(data)[-1]

        self.current_regime = state
        self.state_probabilities = probabilities
        self.current_probability = float(probabilities[state])

        return state
    
    def get_regime_confidence(self):
        return self.current_probability
    
    def get_state_probabilities(self):
        return self.state_probabilities
    
    def get_state_variances(self):
        if self.model is None:
            return None

        covars = self.model.covars_

        if covars.ndim == 3:
            return np.trace(covars, axis1=1, axis2=2)

        return np.sum(covars, axis=1)
    
    def get_regime_name(self):
        # This is a heuristic. We usually associate higher variance states with 'Trending'
        if self.model is None:
            return "Unknown"
        
        state_vars = self.get_state_variances()
        trending_state = np.argmax(state_vars)
        
        return "Trending" if self.current_regime == trending_state else "Ranging"

