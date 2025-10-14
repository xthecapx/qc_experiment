import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
import lmfit
from lmfit.models import (
    ExponentialModel, PowerLawModel, LinearModel, QuadraticModel,
    PolynomialModel, GaussianModel, LorentzianModel, VoigtModel,
    ExponentialGaussianModel, SkewedGaussianModel, BreitWignerModel,
    LognormalModel, DampedOscillatorModel, ExpressionModel
)
import warnings
warnings.filterwarnings('ignore')


class CircuitDepthModels:
    """
    A class for fitting and comparing different models to circuit depth vs error rate data.
    Provides various non-linear models that are more appropriate than linear regression
    for quantum circuit error accumulation patterns.
    """
    
    def __init__(self):
        self.models = {}
        self.fitted_params = {}
        self.model_metrics = {}
        
        # Define all model functions
        self._define_models()
    
    def _define_models(self):
        """Define all the model functions using existing libraries"""
        
        # Store models with their types and configurations
        self.models = {
            # LMFIT built-in models
            'exponential': {
                'type': 'lmfit',
                'model_class': ExponentialModel,
                'description': 'Exponential: amplitude * exp(-x/decay)'
            },
            'power_law': {
                'type': 'lmfit', 
                'model_class': PowerLawModel,
                'description': 'Power Law: amplitude * x^exponent'
            },
            'linear': {
                'type': 'lmfit',
                'model_class': LinearModel,
                'description': 'Linear: slope * x + intercept'
            },
            'quadratic': {
                'type': 'lmfit',
                'model_class': QuadraticModel,
                'description': 'Quadratic: a * x^2 + b * x + c'
            },
            'polynomial_3': {
                'type': 'lmfit',
                'model_class': PolynomialModel,
                'degree': 3,
                'description': 'Polynomial (degree 3): c0 + c1*x + c2*x^2 + c3*x^3'
            },
            'polynomial_4': {
                'type': 'lmfit',
                'model_class': PolynomialModel,
                'degree': 4,
                'description': 'Polynomial (degree 4): c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4'
            },
            'polynomial_5': {
                'type': 'lmfit',
                'model_class': PolynomialModel,
                'degree': 5,
                'description': 'Polynomial (degree 5): c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5'
            },
            
            # Additional lmfit built-in models for comparison
            'gaussian': {
                'type': 'lmfit',
                'model_class': GaussianModel,
                'description': 'Gaussian: amplitude * exp(-(x-center)²/(2*sigma²))'
            },
            'lorentzian': {
                'type': 'lmfit',
                'model_class': LorentzianModel,
                'description': 'Lorentzian: amplitude / (1 + ((x-center)/sigma)²)'
            },
            'voigt': {
                'type': 'lmfit',
                'model_class': VoigtModel,
                'description': 'Voigt: Convolution of Gaussian and Lorentzian'
            },
            'lognormal': {
                'type': 'lmfit',
                'model_class': LognormalModel,
                'description': 'Log-Normal: amplitude * exp(-(ln(x/center))²/(2*sigma²))'
            },
            'exponential_gaussian': {
                'type': 'lmfit',
                'model_class': ExponentialGaussianModel,
                'description': 'Exponential Gaussian: Gaussian convoluted with exponential'
            },
            'skewed_gaussian': {
                'type': 'lmfit',
                'model_class': SkewedGaussianModel,
                'description': 'Skewed Gaussian: Gaussian with asymmetry parameter'
            },
            'breit_wigner': {
                'type': 'lmfit',
                'model_class': BreitWignerModel,
                'description': 'Breit-Wigner: amplitude / ((x-center)² + (sigma/2)²)'
            },
            'damped_oscillator': {
                'type': 'lmfit',
                'model_class': DampedOscillatorModel,
                'description': 'Damped Oscillator: amplitude * exp(-x/decay) * sin(frequency*x + phase)'
            },
            
            # Custom expressions using lmfit ExpressionModel
            'logistic': {
                'type': 'lmfit_expression',
                'expression': 'L / (1 + exp(-k * (x - x0)))',
                'parameters': {'L': {'value': 1.0, 'min': 0}, 'k': {'value': 0.1, 'min': 0}, 'x0': {'value': 20, 'min': 0}},
                'description': 'Logistic: L/(1+exp(-k*(x-x0)))'
            },
            'exponential_saturation': {
                'type': 'lmfit_expression',
                'expression': 'a * (1 - exp(-b * x)) + c',
                'parameters': {'a': {'value': 0.8, 'min': 0, 'max': 2}, 'b': {'value': 0.1, 'min': 0, 'max': 1}, 'c': {'value': 0.1, 'min': 0, 'max': 1}},
                'description': 'Exponential Saturation: a*(1-exp(-b*x))+c'
            },
            'hill_equation': {
                'type': 'lmfit_expression',
                'expression': '(a * x**n) / (b**n + x**n)',
                'parameters': {'a': {'value': 1.0, 'min': 0, 'max': 2}, 'b': {'value': 10.0, 'min': 0, 'max': 100}, 'n': {'value': 2.0, 'min': 0.1, 'max': 5}},
                'description': 'Hill Equation: (a*x^n)/(b^n+x^n)'
            },
            'michaelis_menten': {
                'type': 'lmfit_expression',
                'expression': '(a * x) / (b + x)',
                'parameters': {'a': {'value': 1.0, 'min': 0, 'max': 2}, 'b': {'value': 10.0, 'min': 0, 'max': 100}},
                'description': 'Michaelis-Menten: (a*x)/(b+x)'
            },
            'gompertz': {
                'type': 'lmfit_expression',
                'expression': 'a * exp(-b * exp(-c * x))',
                'parameters': {'a': {'value': 1.0, 'min': 0, 'max': 2}, 'b': {'value': 1.0, 'min': 0, 'max': 10}, 'c': {'value': 0.1, 'min': 0, 'max': 1}},
                'description': 'Gompertz: a*exp(-b*exp(-c*x))'
            },
            'quantum_error': {
                'type': 'lmfit_expression',
                'expression': '1 - (1 - p)**x + offset',
                'parameters': {'p': {'value': 0.1, 'min': 0, 'max': 0.5}, 'offset': {'value': 0.0, 'min': -0.5, 'max': 0.5}},
                'description': 'Quantum Error: 1-(1-p)^x+offset'
            },
            'modified_exponential': {
                'type': 'lmfit_expression',
                'expression': '1 / (1 - a * exp(-b * x) + 1e-10)',  # Added epsilon to avoid division by zero
                'parameters': {'a': {'value': 0.5, 'min': 0, 'max': 0.99}, 'b': {'value': 0.1, 'min': 0, 'max': 1}},
                'description': 'Modified Exponential: 1/(1-a*exp(-b*x))'
            },
            'power_saturation': {
                'type': 'lmfit_expression',
                'expression': 'a * x**b / (1 + c * x**b)',
                'parameters': {'a': {'value': 1.0, 'min': 0, 'max': 2}, 'b': {'value': 1.0, 'min': 0, 'max': 3}, 'c': {'value': 1.0, 'min': 0, 'max': 10}},
                'description': 'Power Saturation: a*x^b/(1+c*x^b)'
            },
            'double_exponential': {
                'type': 'lmfit_expression',
                'expression': 'a * exp(b * x) + c * exp(d * x)',
                'parameters': {'a': {'value': 0.5, 'min': -2, 'max': 2}, 'b': {'value': 0.1, 'min': -1, 'max': 1}, 'c': {'value': 0.3, 'min': -2, 'max': 2}, 'd': {'value': 0.05, 'min': -1, 'max': 1}},
                'description': 'Double Exponential: a*exp(b*x)+c*exp(d*x)'
            },
            'weibull': {
                'type': 'lmfit_expression',
                'expression': 'a * (1 - exp(-(x/b)**c))',
                'parameters': {'a': {'value': 1.0, 'min': 0, 'max': 2}, 'b': {'value': 20.0, 'min': 1, 'max': 100}, 'c': {'value': 2.0, 'min': 0.1, 'max': 5}},
                'description': 'Weibull CDF: a*(1-exp(-(x/b)^c))'
            },
            'stretched_exponential': {
                'type': 'lmfit_expression',
                'expression': 'a * (1 - exp(-(x/tau)**beta))',
                'parameters': {'a': {'value': 1.0, 'min': 0, 'max': 2}, 'tau': {'value': 20.0, 'min': 1, 'max': 100}, 'beta': {'value': 0.5, 'min': 0.1, 'max': 2}},
                'description': 'Stretched Exponential: a*(1-exp(-(x/tau)^beta))'
            },
            'sigmoid_exponential': {
                'type': 'lmfit_expression',
                'expression': 'a / (1 + exp(-k * x)) + b * exp(-c * x)',
                'parameters': {'a': {'value': 1.0, 'min': 0, 'max': 2}, 'k': {'value': 0.1, 'min': 0, 'max': 1}, 'b': {'value': 0.1, 'min': 0, 'max': 1}, 'c': {'value': 0.1, 'min': 0, 'max': 1}},
                'description': 'Sigmoid + Exponential: a/(1+exp(-k*x)) + b*exp(-c*x)'
            }
        }
    
    def _calculate_statsmodels_metrics(self, x_data, y_data, y_pred, n_params):
        """
        Calculate comprehensive model diagnostics using statsmodels
        
        Args:
            x_data: Independent variable
            y_data: Observed dependent variable
            y_pred: Predicted dependent variable
            n_params: Number of model parameters
            
        Returns:
            dict: Comprehensive diagnostics including AIC, BIC, and statistical tests
        """
        try:
            n = len(y_data)
            residuals = y_data - y_pred
            
            # Calculate log-likelihood more robustly
            sse = np.sum(residuals**2)
            mse = sse / n
            
            # Avoid log(0) by adding small epsilon
            if mse <= 0:
                mse = 1e-10
            
            # Log-likelihood for normal distribution
            log_likelihood = -n/2 * np.log(2 * np.pi * mse) - sse / (2 * mse)
            
            # Information criteria
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(n) - 2 * log_likelihood
            
            # Adjusted R²
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1) if n > n_params + 1 else r2
            
            # Additional diagnostics
            diagnostics = {
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'adj_r2': adj_r2,
                'n_observations': n,
                'n_parameters': n_params,
                'degrees_freedom': n - n_params,
                'residual_std_error': np.sqrt(mse)
            }
            
            # Statistical tests (if we have enough data points)
            if n > 10:
                try:
                    # Create a simple OLS model for diagnostic tests
                    # Use polynomial features to approximate the relationship
                    X_design = sm.add_constant(x_data)
                    if X_design.shape[1] < n:  # Ensure we don't have more features than observations
                        ols_model = sm.OLS(y_data, X_design).fit()
                        
                        # Durbin-Watson test for autocorrelation
                        dw_stat = durbin_watson(residuals)
                        diagnostics['durbin_watson'] = dw_stat
                        
                        # Heteroscedasticity tests (if possible)
                        if len(residuals) > X_design.shape[1]:
                            try:
                                # Breusch-Pagan test
                                bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_design)
                                diagnostics['breusch_pagan_stat'] = bp_stat
                                diagnostics['breusch_pagan_pvalue'] = bp_pvalue
                            except:
                                pass
                            
                            try:
                                # White test
                                white_stat, white_pvalue, _, _ = het_white(residuals, X_design)
                                diagnostics['white_stat'] = white_stat
                                diagnostics['white_pvalue'] = white_pvalue
                            except:
                                pass
                        
                except Exception:
                    # If statistical tests fail, continue without them
                    pass
            
            return diagnostics
            
        except Exception as e:
            # Return basic metrics if advanced diagnostics fail
            return {
                'aic': 2 * n_params - 2 * (-n/2 * np.log(2 * np.pi * np.var(y_data)) - n/2),
                'bic': n_params * np.log(n) - 2 * (-n/2 * np.log(2 * np.pi * np.var(y_data)) - n/2),
                'log_likelihood': -n/2 * np.log(2 * np.pi * np.var(y_data)) - n/2,
                'adj_r2': 0,
                'n_observations': n,
                'n_parameters': n_params,
                'degrees_freedom': n - n_params,
                'residual_std_error': np.sqrt(np.var(y_data)),
                'error': str(e)
            }

    def fit_model(self, x_data, y_data, model_name):
        """
        Fit a specific model to the data using appropriate library
        
        Args:
            x_data: Independent variable (circuit depth)
            y_data: Dependent variable (error rate)
            model_name: Name of the model to fit
            
        Returns:
            dict: Fitted parameters and metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model_info = self.models[model_name]
        
        try:
            if model_info['type'] == 'lmfit':
                return self._fit_lmfit_model(x_data, y_data, model_name, model_info)
            elif model_info['type'] == 'lmfit_expression':
                return self._fit_lmfit_expression_model(x_data, y_data, model_name, model_info)
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")
                
        except Exception as e:
            # If fitting fails, return error info
            results = {
                'parameters': None,
                'r2': -np.inf,
                'mse': np.inf,
                'rmse': np.inf,
                'aic': np.inf,
                'bic': np.inf,
                'predictions': None,
                'success': False,
                'error': str(e),
                'description': model_info['description']
            }
            
            self.fitted_params[model_name] = results
            return results
    
    def _fit_lmfit_model(self, x_data, y_data, model_name, model_info):
        """Fit model using lmfit built-in models"""
        model_class = model_info['model_class']
        
        # Handle PolynomialModel with degree parameter
        if model_class == PolynomialModel:
            degree = model_info.get('degree', 2)
            model = model_class(degree=degree)
        else:
            model = model_class()
            
        params = model.guess(y_data, x=x_data)
        result = model.fit(y_data, params, x=x_data)
        
        # Extract parameters
        param_values = [result.params[name].value for name in result.params]
        y_pred = result.best_fit
        
        # Calculate metrics
        r2 = r2_score(y_data, y_pred)
        mse = mean_squared_error(y_data, y_pred)
        rmse = np.sqrt(mse)
        
        # Use statsmodels for proper model comparison metrics
        statsmodels_results = self._calculate_statsmodels_metrics(x_data, y_data, y_pred, len(param_values))
        
        results = {
            'parameters': param_values,
            'parameter_names': list(result.params.keys()),
            'lmfit_result': result,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'aic': statsmodels_results['aic'],
            'bic': statsmodels_results['bic'],
            'predictions': y_pred,
            'success': True,
            'description': model_info['description'],
            'residuals': y_data - y_pred,
            'statsmodels_diagnostics': statsmodels_results
        }
        
        self.fitted_params[model_name] = results
        return results
    
    def _fit_lmfit_expression_model(self, x_data, y_data, model_name, model_info):
        """Fit model using lmfit ExpressionModel"""
        model = ExpressionModel(model_info['expression'])
        params = model.make_params()
        
        # Set parameter constraints
        for param_name, param_config in model_info['parameters'].items():
            if param_name in params:
                params[param_name].set(**param_config)
        
        result = model.fit(y_data, params, x=x_data)
        
        # Extract parameters
        param_values = [result.params[name].value for name in result.params]
        y_pred = result.best_fit
        
        # Calculate metrics
        r2 = r2_score(y_data, y_pred)
        mse = mean_squared_error(y_data, y_pred)
        rmse = np.sqrt(mse)
        
        # Use statsmodels for proper model comparison metrics
        statsmodels_results = self._calculate_statsmodels_metrics(x_data, y_data, y_pred, len(param_values))
        
        results = {
            'parameters': param_values,
            'parameter_names': list(result.params.keys()),
            'lmfit_result': result,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'aic': statsmodels_results['aic'],
            'bic': statsmodels_results['bic'],
            'predictions': y_pred,
            'success': True,
            'description': model_info['description'],
            'residuals': y_data - y_pred,
            'statsmodels_diagnostics': statsmodels_results
        }
        
        self.fitted_params[model_name] = results
        return results
    
    
    def fit_all_models(self, x_data, y_data):
        """
        Fit all available models to the data
        
        Args:
            x_data: Independent variable (circuit depth)
            y_data: Dependent variable (error rate)
            
        Returns:
            dict: Results for all models
        """
        results = {}
        
        for model_name in self.models.keys():
            print(f"Fitting {model_name}...")
            results[model_name] = self.fit_model(x_data, y_data, model_name)
        
        self.model_metrics = results
        return results
    
    def get_comparison_table(self):
        """
        Create a comparison table of all fitted models
        
        Returns:
            pandas.DataFrame: Comparison table with metrics
        """
        if not self.model_metrics:
            raise ValueError("No models have been fitted yet. Call fit_all_models() first.")
        
        comparison_data = []
        
        for model_name, results in self.model_metrics.items():
            if results['success']:
                # Extract statsmodels diagnostics
                diag = results.get('statsmodels_diagnostics', {})
                comparison_data.append({
                    'Model': model_name,
                    'Description': results['description'],
                    'Success': results['success'],
                    'R²': results['r2'],
                    'Adj R²': diag.get('adj_r2', 'N/A'),
                    'RMSE': results['rmse'],
                    'AIC': results['aic'],
                    'BIC': results['bic'],
                    'Log-Likelihood': diag.get('log_likelihood', 'N/A'),
                    'Durbin-Watson': diag.get('durbin_watson', 'N/A'),
                    'BP p-value': diag.get('breusch_pagan_pvalue', 'N/A'),
                    'Parameters': len(results['parameters']) if results['parameters'] is not None else 'Failed',
                    'Degrees of Freedom': diag.get('degrees_freedom', 'N/A')
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'Description': results['description'],
                    'Success': results['success'],
                    'R²': 'Failed',
                    'Adj R²': 'Failed',
                    'RMSE': 'Failed',
                    'AIC': 'Failed',
                    'BIC': 'Failed',
                    'Log-Likelihood': 'Failed',
                    'Durbin-Watson': 'Failed',
                    'BP p-value': 'Failed',
                    'Parameters': 'Failed',
                    'Degrees of Freedom': 'Failed'
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by R² (descending) for successful models
        successful_models = df[df['Success'] == True].copy()
        failed_models = df[df['Success'] == False].copy()
        
        if not successful_models.empty:
            successful_models = successful_models.sort_values('R²', ascending=False)
        
        # Combine successful and failed models
        df_sorted = pd.concat([successful_models, failed_models], ignore_index=True)
        
        return df_sorted
    
    def get_best_model(self, criterion='composite'):
        """
        Get the best model based on specified criterion with proper model selection
        
        Args:
            criterion: 'r2', 'aic', 'bic', 'composite', or 'cross_validation'
            
        Returns:
            tuple: (model_name, model_results, selection_rationale)
        """
        if not self.model_metrics:
            raise ValueError("No models have been fitted yet. Call fit_all_models() first.")
        
        successful_models = {name: results for name, results in self.model_metrics.items() 
                           if results['success']}
        
        if not successful_models:
            raise ValueError("No models were successfully fitted.")
        
        if criterion == 'r2':
            best_model = max(successful_models.items(), key=lambda x: x[1]['r2'])
            rationale = f"Selected based on highest R² = {best_model[1]['r2']:.4f}"
            
        elif criterion == 'aic':
            best_model = min(successful_models.items(), key=lambda x: x[1]['aic'])
            rationale = f"Selected based on lowest AIC = {best_model[1]['aic']:.2f}"
            
        elif criterion == 'bic':
            best_model = min(successful_models.items(), key=lambda x: x[1]['bic'])
            rationale = f"Selected based on lowest BIC = {best_model[1]['bic']:.2f}"
            
        elif criterion == 'composite':
            # Composite scoring considering multiple criteria
            best_model, rationale = self._select_best_model_composite(successful_models)
            
        elif criterion == 'cross_validation':
            # Use cross-validation for model selection
            best_model, rationale = self._select_best_model_cv(successful_models)
            
        else:
            raise ValueError("Criterion must be 'r2', 'aic', 'bic', 'composite', or 'cross_validation'")
        
        return best_model[0], best_model[1], rationale
    
    def _select_best_model_composite(self, successful_models):
        """
        Select best model using composite scoring with multiple criteria
        """
        # Normalize metrics for composite scoring
        r2_values = [results['r2'] for results in successful_models.values()]
        aic_values = [results['aic'] for results in successful_models.values()]
        bic_values = [results['bic'] for results in successful_models.values()]
        
        # Get adjusted R² values
        adj_r2_values = []
        for results in successful_models.values():
            diag = results.get('statsmodels_diagnostics', {})
            adj_r2_values.append(diag.get('adj_r2', results['r2']))
        
        # Normalize to 0-1 scale
        r2_norm = [(r2 - min(r2_values)) / (max(r2_values) - min(r2_values)) if max(r2_values) > min(r2_values) else 0.5 
                   for r2 in r2_values]
        adj_r2_norm = [(adj_r2 - min(adj_r2_values)) / (max(adj_r2_values) - min(adj_r2_values)) if max(adj_r2_values) > min(adj_r2_values) else 0.5 
                       for adj_r2 in adj_r2_values]
        
        # For AIC/BIC, lower is better, so we invert
        aic_norm = [(max(aic_values) - aic) / (max(aic_values) - min(aic_values)) if max(aic_values) > min(aic_values) else 0.5 
                    for aic in aic_values]
        bic_norm = [(max(bic_values) - bic) / (max(bic_values) - min(bic_values)) if max(bic_values) > min(bic_values) else 0.5 
                    for bic in bic_values]
        
        # Composite score: weighted combination
        # Emphasize adjusted R² and BIC (penalizes complexity more than AIC)
        weights = {'adj_r2': 0.4, 'bic': 0.3, 'r2': 0.2, 'aic': 0.1}
        
        composite_scores = {}
        for i, (name, results) in enumerate(successful_models.items()):
            score = (weights['adj_r2'] * adj_r2_norm[i] + 
                    weights['bic'] * bic_norm[i] + 
                    weights['r2'] * r2_norm[i] + 
                    weights['aic'] * aic_norm[i])
            
            # Penalty for overly complex models (>4 parameters)
            n_params = len(results['parameters']) if results['parameters'] is not None else 0
            if n_params > 4:
                score *= 0.9  # 10% penalty
            
            composite_scores[name] = score
        
        best_name = max(composite_scores.items(), key=lambda x: x[1])[0]
        best_model = (best_name, successful_models[best_name])
        
        # Create detailed rationale
        best_results = successful_models[best_name]
        diag = best_results.get('statsmodels_diagnostics', {})
        rationale = (f"Selected using composite scoring (score={composite_scores[best_name]:.3f}): "
                    f"R²={best_results['r2']:.4f}, Adj R²={diag.get('adj_r2', 'N/A'):.4f}, "
                    f"AIC={best_results['aic']:.2f}, BIC={best_results['bic']:.2f}")
        
        return best_model, rationale
    
    def _select_best_model_cv(self, successful_models):
        """
        Select best model using cross-validation (simplified version)
        Note: This is a placeholder for more sophisticated CV implementation
        """
        # For now, use BIC as proxy for cross-validation performance
        # BIC approximates leave-one-out cross-validation for large samples
        best_model = min(successful_models.items(), key=lambda x: x[1]['bic'])
        rationale = f"Selected using BIC (approximates CV): BIC = {best_model[1]['bic']:.2f}"
        
        return best_model, rationale
    
    def predict(self, model_name, x_values):
        """
        Make predictions using a fitted model
        
        Args:
            model_name: Name of the fitted model
            x_values: Values to predict for
            
        Returns:
            numpy.array: Predictions
        """
        if model_name not in self.fitted_params:
            raise ValueError(f"Model {model_name} has not been fitted yet.")
        
        if not self.fitted_params[model_name]['success']:
            raise ValueError(f"Model {model_name} fitting failed.")
        
        results = self.fitted_params[model_name]
        model_info = self.models[model_name]
        
        if model_info['type'] in ['lmfit', 'lmfit_expression']:
            # Use lmfit result to predict
            lmfit_result = results['lmfit_result']
            return lmfit_result.eval(x=x_values)
            
        elif model_info['type'] == 'sklearn':
            # Use sklearn model to predict
            sklearn_model = results['sklearn_model']
            X = x_values.reshape(-1, 1)
            return sklearn_model.predict(X)
            
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
    
    def get_model_equation(self, model_name):
        """
        Get the fitted equation for a model with parameter values
        
        Args:
            model_name: Name of the fitted model
            
        Returns:
            str: Formatted equation string
        """
        if model_name not in self.fitted_params:
            raise ValueError(f"Model {model_name} has not been fitted yet.")
        
        if not self.fitted_params[model_name]['success']:
            return f"Model {model_name} fitting failed"
        
        results = self.fitted_params[model_name]
        params = results['parameters']
        description = results['description']
        
        # Format parameters with names if available
        if 'parameter_names' in results and results['parameter_names']:
            param_names = results['parameter_names']
            param_str = ', '.join([f'{name}={val:.4f}' for name, val in zip(param_names, params)])
        else:
            param_str = ', '.join([f'{p:.4f}' for p in params])
        
        return f"{description} | Parameters: [{param_str}]"
    
    def get_detailed_diagnostics(self, model_name):
        """
        Get detailed statsmodels diagnostics for a specific model
        
        Args:
            model_name: Name of the fitted model
            
        Returns:
            dict: Detailed diagnostic information
        """
        if model_name not in self.fitted_params:
            raise ValueError(f"Model {model_name} has not been fitted yet.")
        
        if not self.fitted_params[model_name]['success']:
            return f"Model {model_name} fitting failed"
        
        results = self.fitted_params[model_name]
        diagnostics = results.get('statsmodels_diagnostics', {})
        
        # Format the diagnostics nicely
        formatted_diagnostics = {
            'Model': model_name,
            'Description': results['description'],
            'Parameters': results['parameters'],
            'R²': results['r2'],
            'Adjusted R²': diagnostics.get('adj_r2', 'N/A'),
            'RMSE': results['rmse'],
            'AIC': results['aic'],
            'BIC': results['bic'],
            'Log-Likelihood': diagnostics.get('log_likelihood', 'N/A'),
            'Residual Standard Error': diagnostics.get('residual_std_error', 'N/A'),
            'Degrees of Freedom': diagnostics.get('degrees_freedom', 'N/A'),
            'Number of Observations': diagnostics.get('n_observations', 'N/A'),
            'Number of Parameters': diagnostics.get('n_parameters', 'N/A')
        }
        
        # Add statistical tests if available
        if 'durbin_watson' in diagnostics:
            formatted_diagnostics['Durbin-Watson Statistic'] = diagnostics['durbin_watson']
            
        if 'breusch_pagan_pvalue' in diagnostics:
            formatted_diagnostics['Breusch-Pagan Test p-value'] = diagnostics['breusch_pagan_pvalue']
            formatted_diagnostics['Heteroscedasticity (BP)'] = 'Detected' if diagnostics['breusch_pagan_pvalue'] < 0.05 else 'Not Detected'
            
        if 'white_pvalue' in diagnostics:
            formatted_diagnostics['White Test p-value'] = diagnostics['white_pvalue']
            formatted_diagnostics['Heteroscedasticity (White)'] = 'Detected' if diagnostics['white_pvalue'] < 0.05 else 'Not Detected'
        
        return formatted_diagnostics
    
    def plot_model_comparison(self, x_data, y_data, top_n=5, save_path=None):
        """
        Plot the top N models for visual comparison
        
        Args:
            x_data: Independent variable data
            y_data: Dependent variable data  
            top_n: Number of top models to plot
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        import matplotlib.pyplot as plt
        
        if not self.model_metrics:
            raise ValueError("No models have been fitted yet. Call fit_all_models() first.")
        
        # Get top N models by composite score
        successful_models = {name: results for name, results in self.model_metrics.items() 
                           if results['success']}
        
        if not successful_models:
            raise ValueError("No models were successfully fitted.")
        
        # Calculate composite scores for ranking
        _, _ = self._select_best_model_composite(successful_models)
        
        # Sort models by R² for now (could use composite score)
        sorted_models = sorted(successful_models.items(), 
                             key=lambda x: x[1]['r2'], reverse=True)[:top_n]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Data points and model fits
        ax1.scatter(x_data, y_data, color='black', s=50, alpha=0.7, 
                   label='Observed Data', zorder=10)
        
        # Generate smooth prediction line
        x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
        
        colors = plt.cm.Set1(np.linspace(0, 1, top_n))
        
        for i, (model_name, results) in enumerate(sorted_models):
            try:
                y_smooth = self.predict(model_name, x_smooth)
                ax1.plot(x_smooth, y_smooth, color=colors[i], linewidth=2,
                        label=f'{model_name} (R²={results["r2"]:.3f})')
            except Exception as e:
                print(f"Warning: Could not plot {model_name}: {e}")
        
        ax1.set_xlabel('Circuit Depth', fontweight='bold')
        ax1.set_ylabel('Error Rate', fontweight='bold')
        ax1.set_title('Model Fits Comparison', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals for best model
        best_name = sorted_models[0][0]
        best_results = sorted_models[0][1]
        residuals = best_results['residuals']
        
        ax2.scatter(x_data, residuals, color=colors[0], alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Circuit Depth', fontweight='bold')
        ax2.set_ylabel('Residuals', fontweight='bold')
        ax2.set_title(f'Residuals: {best_name}', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_model_selection_report(self):
        """
        Generate a comprehensive model selection report
        
        Returns:
            str: Formatted report with model selection rationale
        """
        if not self.model_metrics:
            raise ValueError("No models have been fitted yet. Call fit_all_models() first.")
        
        successful_models = {name: results for name, results in self.model_metrics.items() 
                           if results['success']}
        
        if not successful_models:
            return "No models were successfully fitted."
        
        report = []
        report.append("="*80)
        report.append("MODEL SELECTION REPORT")
        report.append("="*80)
        
        # Get best models by different criteria
        criteria = ['composite', 'bic', 'aic', 'r2']
        best_models = {}
        
        for criterion in criteria:
            try:
                name, results, rationale = self.get_best_model(criterion)
                best_models[criterion] = (name, results, rationale)
            except Exception as e:
                best_models[criterion] = None
        
        report.append("\nBEST MODELS BY DIFFERENT CRITERIA:")
        report.append("-" * 50)
        
        for criterion, model_info in best_models.items():
            if model_info:
                name, results, rationale = model_info
                report.append(f"\n{criterion.upper()}: {name}")
                report.append(f"  {rationale}")
            else:
                report.append(f"\n{criterion.upper()}: Failed to determine")
        
        # Model complexity analysis
        report.append("\n\nMODEL COMPLEXITY ANALYSIS:")
        report.append("-" * 50)
        
        complexity_models = []
        for name, results in successful_models.items():
            n_params = len(results['parameters']) if results['parameters'] is not None else 0
            diag = results.get('statsmodels_diagnostics', {})
            adj_r2 = diag.get('adj_r2', results['r2'])
            
            complexity_models.append({
                'name': name,
                'n_params': n_params,
                'r2': results['r2'],
                'adj_r2': adj_r2,
                'bic': results['bic'],
                'complexity_penalty': results['r2'] - adj_r2
            })
        
        # Sort by complexity penalty (overfitting indicator)
        complexity_models.sort(key=lambda x: x['complexity_penalty'])
        
        report.append(f"{'Model':<20} {'Params':<8} {'R²':<8} {'Adj R²':<8} {'BIC':<10} {'Penalty':<8}")
        report.append("-" * 70)
        
        for model in complexity_models[:10]:  # Top 10
            report.append(f"{model['name']:<20} {model['n_params']:<8} "
                         f"{model['r2']:<8.4f} {model['adj_r2']:<8.4f} "
                         f"{model['bic']:<10.2f} {model['complexity_penalty']:<8.4f}")
        
        # Recommendations
        report.append("\n\nRECOMMENDations:")
        report.append("-" * 50)
        
        if best_models['composite']:
            composite_name = best_models['composite'][0]
            report.append(f"🏆 RECOMMENDED MODEL: {composite_name}")
            report.append(f"   Rationale: {best_models['composite'][2]}")
        
        # Check for overfitting warnings
        high_complexity = [m for m in complexity_models if m['complexity_penalty'] > 0.1]
        if high_complexity:
            report.append(f"\n⚠️  OVERFITTING WARNING: {len(high_complexity)} models show high complexity penalty")
            for model in high_complexity[:3]:
                report.append(f"   - {model['name']}: penalty = {model['complexity_penalty']:.4f}")
        
        return "\n".join(report)
