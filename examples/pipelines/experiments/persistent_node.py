from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import json
from pathlib import Path
import hashlib
import numpy as np


class PersistentNode(BaseEstimator):
    def __init__(self, save_path="models/unnamed.joblib"):
        self.save_path = save_path
        self._path = Path(save_path)
        self._meta_path = self._path.with_suffix('.meta.json')
        self.fitted_ = False
        self.model = None

    def _save_file(self, obj, input_hash=None):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, self._path)
        if input_hash is not None:
            self._meta_path.write_text(json.dumps({'hash': input_hash}))
        self.fitted_ = True

    def save_model_with_hash(self, obj, data):
        input_hash = self.hash_input(data)
        self._save_file(obj, input_hash=input_hash)

    def load(self):
        obj = joblib.load(self._path)
        self.fitted_ = True
        return obj

    def exists(self):
        return self._path.exists() and self._meta_path.exists()

    def get_stored_hash(self):
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text()).get('hash')
        return None

    def hash_input(self, data):
        m = hashlib.sha256()
        if isinstance(data, tuple):
            for arr in data:
                m.update(np.ascontiguousarray(arr).data)
        else:
            m.update(np.ascontiguousarray(data).data)
        return m.hexdigest()

    def check_fitted_or_load(self):
        if self.fitted_:
            return
        if self.exists():
            loaded = self.load()
            self.set_model(loaded)
        else:
            raise ValueError("Model not fitted or not saved.")

    def set_model(self, obj):
        self.model = obj

    def get_model(self):
        return self.model

    def load_if_cached(self, data):
        input_hash = self.hash_input(data)
        if self.exists() and self.get_stored_hash() == input_hash:
            self.set_model(self.load())
            return True
        return False

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if deep and hasattr(self.model, "get_params"):
            for k, v in self.model.get_params(deep=True).items():
                params[f"model__{k}"] = v
        return params

    def set_params(self, **params):
        model_params = {k[7:]: v for k, v in params.items() if k.startswith("model__")}
        node_params = {k: v for k, v in params.items() if not k.startswith("model__")}
        if node_params:
            super().set_params(**node_params)
        if model_params and self.model is not None:
            self.model.set_params(**model_params)
        return self



from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from pipefunc import pipefunc, Pipeline
import numpy as np
import inspect
from typing import Dict, Any, Optional

class PipefuncSklearnWrapper(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for pipefunc Pipeline objects.

    This wrapper allows pipefunc pipelines to be used with scikit-learn tools
    like GridSearchCV, cross_val_score, etc.
    """

    def __init__(self, pipeline: Pipeline, prediction_output: str = "predictions", **kwargs):
        """
        Initialize the wrapper.

        Parameters:
        -----------
        pipeline : pipefunc.Pipeline
            The pipefunc pipeline to wrap
        prediction_output : str
            Name of the pipeline output that contains predictions
        **kwargs : dict
            Additional parameters that can be tuned via GridSearchCV
        """
        self.pipeline = pipeline
        self.prediction_output = prediction_output

        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Keep track of parameter names for get_params/set_params
        self._param_names = set(kwargs.keys())
        self._param_names.add('pipeline')
        self._param_names.add('prediction_output')

        # Training data storage
        self.X_train_ = None
        self.y_train_ = None
        self.is_fitted_ = False

    def _rebuild_pipeline_with_params(self) -> Pipeline:
        """
        Rebuild the pipeline with current parameters.

        This method attempts to inject current parameters into pipeline functions
        that accept them as arguments.
        """
        # For now, return the original pipeline
        # In a more sophisticated implementation, you would:
        # 1. Extract functions from the original pipeline
        # 2. Create new functions with updated parameters
        # 3. Rebuild the pipeline
        return self.pipeline

    def _validate_pipeline_output(self, result):
        """Validate that the pipeline returns the expected output."""
        if isinstance(result, dict):
            if self.prediction_output not in result:
                available_outputs = list(result.keys())
                raise ValueError(
                    f"Pipeline output '{self.prediction_output}' not found. "
                    f"Available outputs: {available_outputs}"
                )
            return result[self.prediction_output]
        else:
            # Assume the result is the direct prediction
            return result

    def fit(self, X, y):
        """
        Fit the pipeline.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Store training data
        self.X_train_ = X.copy() if hasattr(X, 'copy') else np.array(X)
        self.y_train_ = y.copy() if hasattr(y, 'copy') else np.array(y)

        # Rebuild pipeline with current parameters
        self.pipeline_ = self._rebuild_pipeline_with_params()

        # Mark as fitted
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """
        Make predictions using the fitted pipeline.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict on

        Returns:
        --------
        predictions : array-like of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("This PipefuncSklearnWrapper instance is not fitted yet.")

        try:
            # Execute the pipeline with training data and test data
            # This assumes the pipeline expects both training and test data
            result = self.pipeline_(
                X=self.X_train_,
                y=self.y_train_,
                X_test=X
            )

            # Extract predictions from result
            predictions = self._validate_pipeline_output(result)

            return predictions

        except Exception as e:
            # Try alternative calling patterns
            try:
                # Maybe the pipeline expects different argument names
                result = self.pipeline_(
                    X_train=self.X_train_,
                    y_train=self.y_train_,
                    X_test=X
                )
                return self._validate_pipeline_output(result)
            except:
                # Try calling with specific output name
                try:
                    result = self.pipeline_(
                        self.prediction_output,
                        X=self.X_train_,
                        y=self.y_train_,
                        X_test=X
                    )
                    return result
                except:
                    raise RuntimeError(
                        f"Failed to execute pipeline for prediction. "
                        f"Original error: {str(e)}"
                    )

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X

        Returns:
        --------
        score : float
            R^2 score
        """
        from sklearn.metrics import r2_score

        predictions = self.predict(X)
        return r2_score(y, predictions)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values
        """
        params = {}

        # Get all stored parameters
        for param_name in self._param_names:
            if hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)

        # If deep=True, try to extract parameters from pipeline functions
        if deep and hasattr(self, 'pipeline') and self.pipeline is not None:
            try:
                # Try to access pipefunc internal structure
                if hasattr(self.pipeline, 'graph'):
                    # NetworkX graph-based approach
                    for node_name in self.pipeline.graph.nodes():
                        node_data = self.pipeline.graph.nodes[node_name]
                        if 'func' in node_data:
                            func = node_data['func']
                            # Extract function parameters
                            self._extract_function_params(func, node_name, params)
            except Exception:
                # If we can't extract deep parameters, continue silently
                pass

        return params

    def _extract_function_params(self, func, node_name, params):
        """Extract parameters from a pipeline function."""
        try:
            # Get the original function if it's wrapped
            original_func = func
            if hasattr(func, '__wrapped__'):
                original_func = func.__wrapped__

            # Get function signature
            sig = inspect.signature(original_func)

            # Extract parameters with defaults
            for param_name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    prefixed_name = f"{node_name}__{param_name}"
                    params[prefixed_name] = param.default
        except Exception:
            # If parameter extraction fails, continue silently
            pass

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters:
        -----------
        **params : dict
            Estimator parameters

        Returns:
        --------
        self : object
            Estimator instance
        """
        valid_params = self.get_params(deep=True)

        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
                # Add to tracked parameters if not already there
                if key not in self._param_names:
                    self._param_names.add(key)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params.keys())}"
                )

        # Reset fitted state if parameters changed
        self.is_fitted_ = False

        return self




# class PipefuncWrapper(BaseEstimator, RegressorMixin):
#     def __init__(self, pipe=None):
#         self.pipe = pipe

#     def fit(self, X, y):
#         self.pipe.clear()  # make sure it's fresh
#         self.pipe.map({
#             "X": X,
#             "y": y
#         })
#         return self

#     def predict(self, X):
#         # Replace inputs, remove cached results downstream
#         self.pipe.clear()
#         self.pipe.map({
#             "X": X
#         })
#         return self.pipe["y_pred"]

#     def get_params(self, deep=True):
#         """Get parameters for this estimator"""
#         params = {}

#         # Get all parameters stored as instance attributes
#         for param_name in self._param_names:
#             if hasattr(self, param_name):
#                 params[param_name] = getattr(self, param_name)

#         # If deep=True and we have a pipeline, try to get function parameters
#         if deep and self.pipe is not None:
#             try:
#                 # pipefunc stores functions in a different way
#                 # Access the internal graph structure
#                 if hasattr(self.pipe, 'graph'):
#                     for node_name in self.pipe.graph.nodes():
#                         node_data = self.pipe.graph.nodes[node_name]
#                         if 'func' in node_data:
#                             func = node_data['func']
#                             # Get function signature parameters
#                             if hasattr(func, '__wrapped__'):  # For decorated functions
#                                 func = func.__wrapped__
#                             sig = inspect.signature(func)
#                             for param_name, param in sig.parameters.items():
#                                 if param.default is not inspect.Parameter.empty:
#                                     params[f"{node_name}__{param_name}"] = param.default
#                 elif hasattr(self.pipe, '_functions'):
#                     # Alternative access pattern
#                     for func in self.pipe._functions:
#                         func_name = getattr(func, 'output_name', func.__name__)
#                         if hasattr(func, '__wrapped__'):
#                             original_func = func.__wrapped__
#                         else:
#                             original_func = func
#                         sig = inspect.signature(original_func)
#                         for param_name, param in sig.parameters.items():
#                             if param.default is not inspect.Parameter.empty:
#                                 params[f"{func_name}__{param_name}"] = param.default
#             except Exception as e:
#                 # If we can't extract deep parameters, just continue
#                 # This ensures compatibility even if pipefunc internals change
#                 pass

#         return params

#     def set_params(self, **params):
#         """Set parameters for this estimator"""
#         valid_params = self.get_params(deep=True)

#         for key, value in params.items():
#             if key in valid_params:
#                 if "__" in key:
#                     # This is a nested parameter (function parameter)
#                     # Store it for use when rebuilding the pipeline
#                     setattr(self, key, value)
#                 else:
#                     # This is a top-level parameter
#                     setattr(self, key, value)
#                     if key not in self._param_names:
#                         self._param_names.add(key)
#             else:
#                 raise ValueError(f"Invalid parameter {key}")