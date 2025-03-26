import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# Add ipywidgets for interactive functionality
try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Note: ipywidgets not found. Interactive features will be disabled.")
    print("To enable, install with: pip install ipywidgets")

class ModelExplorer:
    """A simple class to explore the quantum circuit success rate prediction model"""
    
    def __init__(self, model_path="target_depth_model.pkl"):
        """Initialize the model explorer with a path to the saved model"""
        self.model_path = model_path
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the regression model from a pickle file"""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}.")
            print("Using the Model 2 equation from equations.txt as fallback:")
            print("log(success_rate + 0.001) = 0.207587 + 0.014493 * circuit_depth - 0.009495 * circuit_size - 0.071598 * circuit_width - 0.347179 * payload_size")
            
            # Create a simple dummy model using the coefficients from the equation
            class DummyModel:
                def __init__(self):
                    self.params = pd.Series({
                        'const': 0.207587,
                        'circuit_depth': 0.014493,
                        'circuit_size': -0.009495,
                        'circuit_width': -0.071598,
                        'payload_size': -0.347179
                    })
                
                def predict(self, X):
                    # Sum the product of each parameter and its coefficient
                    result = 0
                    for col in X.columns:
                        result += X[col].values[0] * self.params[col]
                    return [result]
            
            return DummyModel()
    
    def predict_success_rate(self, circuit_depth, circuit_size, circuit_width, payload_size):
        """Predict success rate for a circuit with given parameters"""
        # Create input features DataFrame
        X = pd.DataFrame({
            'circuit_depth': [circuit_depth],
            'circuit_size': [circuit_size],
            'circuit_width': [circuit_width],
            'payload_size': [payload_size],
            'const': [1.0]  # Add constant term
        })
        
        # Ensure columns are in the right order for the model
        model_params = self.model.params.index.tolist()
        X = X[model_params]
        
        # Make prediction (log-transformed)
        log_pred = self.model.predict(X)[0]
        
        # Transform back to original scale
        success_rate = np.exp(log_pred) - 0.001
        
        # Clip to valid range [0, 1]
        success_rate = np.clip(success_rate, 0, 1)
        
        return success_rate
    
    def visualize_prediction(self, circuit_depth, circuit_size, circuit_width, payload_size):
        """Visualize the prediction for a given set of parameters"""
        # Get the prediction
        success_rate = self.predict_success_rate(circuit_depth, circuit_size, circuit_width, payload_size)
        
        # Display the parameters and prediction
        print(f"Circuit Parameters:")
        print(f"- Depth: {circuit_depth}")
        print(f"- Size: {circuit_size}")
        print(f"- Width: {circuit_width}")
        print(f"- Payload Size: {payload_size}")
        print(f"\nPredicted Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
        
        # Create a simple visualization
        plt.figure(figsize=(8, 5))
        plt.bar(['Success', 'Failure'], [success_rate, 1-success_rate], color=['green', 'red'])
        plt.title(f'Predicted Success Rate', fontsize=14)
        plt.ylabel('Probability', fontsize=12)
        plt.ylim(0, 1)
        
        # Add text labels
        plt.text(0, success_rate/2, f"{success_rate*100:.2f}%", ha='center', fontsize=12, 
                 color='white' if success_rate > 0.3 else 'black')
        plt.text(1, (1-success_rate)/2, f"{(1-success_rate)*100:.2f}%", ha='center', fontsize=12, 
                 color='white' if (1-success_rate) > 0.3 else 'black')
        
        plt.tight_layout()
        plt.show()
    
    def parameter_impact_plot(self, parameter_to_vary, fixed_params=None):
        """Plot the impact of varying a single parameter on success rate"""
        # Default parameter values if not provided
        if fixed_params is None:
            fixed_params = {
                'circuit_depth': 20,
                'circuit_size': 20,
                'circuit_width': 5,
                'payload_size': 3
            }
        
        # Parameter ranges to explore
        ranges = {
            'circuit_depth': range(5, 51, 2),
            'circuit_size': range(5, 101, 5),
            'circuit_width': range(2, 21, 1),
            'payload_size': range(1, 11, 1)
        }
        
        # Calculate success rates for different values of the parameter
        x_values = list(ranges[parameter_to_vary])
        success_rates = []
        
        for val in x_values:
            params = fixed_params.copy()
            params[parameter_to_vary] = val
            success_rates.append(self.predict_success_rate(
                params['circuit_depth'], 
                params['circuit_size'], 
                params['circuit_width'], 
                params['payload_size']
            ))
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, success_rates, 'o-', linewidth=2, markersize=6)
        plt.xlabel(parameter_to_vary.replace('_', ' ').title(), fontsize=12)
        plt.ylabel('Predicted Success Rate', fontsize=12)
        plt.title(f'Impact of {parameter_to_vary.replace("_", " ").title()} on Success Rate', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add threshold line at 0.5 success rate
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Success Rate')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return x_values, success_rates
    
    def create_heatmap(self, param_x='circuit_depth', param_y='payload_size', fixed_params=None):
        """Create a heatmap showing how two parameters interact to affect success rate"""
        # Default parameter values if not provided
        if fixed_params is None:
            fixed_params = {
                'circuit_depth': 20,
                'circuit_size': 20,
                'circuit_width': 5,
                'payload_size': 3
            }
        
        # Parameter ranges to explore
        ranges = {
            'circuit_depth': np.arange(5, 51, 5),
            'circuit_size': np.arange(5, 101, 10),
            'circuit_width': np.arange(2, 21, 2),
            'payload_size': np.arange(1, 11, 1)
        }
        
        # Create meshgrid for parameters
        x_values = ranges[param_x]
        y_values = ranges[param_y]
        X, Y = np.meshgrid(x_values, y_values)
        
        # Calculate success rates for each combination
        Z = np.zeros_like(X, dtype=float)
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                params = fixed_params.copy()
                params[param_x] = X[i, j]
                params[param_y] = Y[i, j]
                Z[i, j] = self.predict_success_rate(
                    params['circuit_depth'], 
                    params['circuit_size'], 
                    params['circuit_width'], 
                    params['payload_size']
                )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        heatmap = plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        plt.colorbar(heatmap, label='Predicted Success Rate')
        
        plt.xlabel(param_x.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(param_y.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Success Rate Heatmap: {param_x.replace("_", " ").title()} vs {param_y.replace("_", " ").title()}', fontsize=14)
        
        # Add contour lines for specific success rates
        contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        C = plt.contour(X, Y, Z, levels=contour_levels, colors='white', alpha=0.8, linewidths=1.5)
        plt.clabel(C, inline=True, fontsize=10, fmt='%.1f')
        
        plt.tight_layout()
        plt.show()
        
        return X, Y, Z
    
    # NEW INTERACTIVE METHODS
    
    def interactive_explorer(self):
        """Create an interactive explorer with sliders for all parameters"""
        if not WIDGETS_AVAILABLE:
            print("Error: ipywidgets not available. Install with: pip install ipywidgets")
            return
        
        # Create output widget for displaying results
        output = widgets.Output()
        
        # Create sliders for each parameter
        depth_slider = widgets.IntSlider(
            value=20, min=5, max=50, step=1, 
            description='Circuit Depth:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )
        
        size_slider = widgets.IntSlider(
            value=20, min=5, max=100, step=5, 
            description='Circuit Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )
        
        width_slider = widgets.IntSlider(
            value=5, min=2, max=20, step=1, 
            description='Circuit Width:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )
        
        payload_slider = widgets.IntSlider(
            value=3, min=1, max=10, step=1, 
            description='Payload Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )
        
        # Link width and payload sliders (optional)
        # Typically circuit_width = payload_size + overhead
        # Uncomment to enable automatic width adjustment
        # def update_width(change):
        #     width_slider.value = change['new'] + 2
        # payload_slider.observe(update_width, names='value')
        
        # Function to update the visualization
        def update_viz(*args):
            with output:
                # Clear previous output
                output.clear_output(wait=True)
                
                # Get current values
                circuit_depth = depth_slider.value
                circuit_size = size_slider.value
                circuit_width = width_slider.value
                payload_size = payload_slider.value
                
                # Calculate success rate
                success_rate = self.predict_success_rate(
                    circuit_depth, circuit_size, circuit_width, payload_size
                )
                
                # Display info
                print(f"Circuit Parameters:")
                print(f"- Depth: {circuit_depth}")
                print(f"- Size: {circuit_size}")
                print(f"- Width: {circuit_width}")
                print(f"- Payload Size: {payload_size}")
                print(f"\nPredicted Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Bar chart (success/failure)
                ax1.bar(['Success', 'Failure'], [success_rate, 1-success_rate], color=['green', 'red'])
                ax1.set_title('Predicted Success Rate', fontsize=14)
                ax1.set_ylabel('Probability', fontsize=12)
                ax1.set_ylim(0, 1)
                
                # Add text labels
                ax1.text(0, success_rate/2, f"{success_rate*100:.2f}%", ha='center', fontsize=12, 
                        color='white' if success_rate > 0.3 else 'black')
                ax1.text(1, (1-success_rate)/2, f"{(1-success_rate)*100:.2f}%", ha='center', fontsize=12, 
                        color='white' if (1-success_rate) > 0.3 else 'black')
                
                # Gauge chart
                gauge_colors = plt.cm.RdYlGn(success_rate)
                ax2.add_patch(plt.matplotlib.patches.Wedge(
                    center=(0.5, 0.5), r=0.4, theta1=180, theta2=180+180*success_rate,
                    color=gauge_colors, alpha=0.8
                ))
                
                # Add background for gauge
                ax2.add_patch(plt.matplotlib.patches.Wedge(
                    center=(0.5, 0.5), r=0.4, theta1=180+180*success_rate, theta2=360,
                    color='lightgray', alpha=0.3
                ))
                
                # Add gauge text
                ax2.text(0.5, 0.4, f"{success_rate*100:.1f}%", 
                        ha='center', va='center', fontsize=18, fontweight='bold')
                ax2.text(0.5, 0.25, "Success Rate", ha='center', va='center', fontsize=14)
                
                # Add success thresholds
                for threshold, label, y_pos in [(0.25, 'Poor', 0.8), (0.5, 'Fair', 0.8), (0.75, 'Good', 0.8)]:
                    ax2.axvline(x=0.5 - 0.4*np.cos(np.pi + threshold*np.pi), 
                              ymin=0.5-0.1, ymax=0.5+0.1, color='black', alpha=0.6)
                
                # Clean up gauge chart
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
                
                plt.tight_layout()
                plt.show()
        
        # Connect sliders to update function
        depth_slider.observe(update_viz, names='value')
        size_slider.observe(update_viz, names='value')
        width_slider.observe(update_viz, names='value')
        payload_slider.observe(update_viz, names='value')
        
        # Create UI layout
        ui = widgets.VBox([
            widgets.HTML(value="<h3>Circuit Success Rate Predictor</h3>"),
            widgets.HBox([widgets.VBox([
                depth_slider, 
                size_slider, 
                width_slider, 
                payload_slider
            ])]),
            output
        ])
        
        # Run initial update
        update_viz()
        
        # Display the UI
        display(ui)
    
    def interactive_comparison(self):
        """Create an interactive explorer that compares two circuit configurations"""
        if not WIDGETS_AVAILABLE:
            print("Error: ipywidgets not available. Install with: pip install ipywidgets")
            return
        
        # Create output widget for displaying results
        output = widgets.Output()
        
        # Circuit A sliders
        depth_a = widgets.IntSlider(value=20, min=5, max=50, step=1, description='Depth A:')
        size_a = widgets.IntSlider(value=20, min=5, max=100, step=5, description='Size A:')
        width_a = widgets.IntSlider(value=5, min=2, max=20, step=1, description='Width A:')
        payload_a = widgets.IntSlider(value=3, min=1, max=10, step=1, description='Payload A:')
        
        # Circuit B sliders
        depth_b = widgets.IntSlider(value=30, min=5, max=50, step=1, description='Depth B:')
        size_b = widgets.IntSlider(value=20, min=5, max=100, step=5, description='Size B:')
        width_b = widgets.IntSlider(value=5, min=2, max=20, step=1, description='Width B:')
        payload_b = widgets.IntSlider(value=3, min=1, max=10, step=1, description='Payload B:')
        
        # Function to update comparison visualization
        def update_comparison(*args):
            with output:
                # Clear previous output
                output.clear_output(wait=True)
                
                # Get current values for both circuits
                circuit_a = {
                    'depth': depth_a.value,
                    'size': size_a.value,
                    'width': width_a.value,
                    'payload': payload_a.value
                }
                
                circuit_b = {
                    'depth': depth_b.value,
                    'size': size_b.value,
                    'width': width_b.value,
                    'payload': payload_b.value
                }
                
                # Calculate success rates
                success_a = self.predict_success_rate(
                    circuit_a['depth'], circuit_a['size'], circuit_a['width'], circuit_a['payload']
                )
                
                success_b = self.predict_success_rate(
                    circuit_b['depth'], circuit_b['size'], circuit_b['width'], circuit_b['payload']
                )
                
                # Display the comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Circuit A
                ax1.bar(['Success', 'Failure'], [success_a, 1-success_a], color=['green', 'red'])
                ax1.set_title(f'Circuit A - Success Rate: {success_a*100:.2f}%', fontsize=14)
                ax1.set_ylim(0, 1)
                ax1.text(0, success_a/2, f"{success_a*100:.2f}%", ha='center', fontsize=12, 
                      color='white' if success_a > 0.3 else 'black')
                ax1.text(1, (1-success_a)/2, f"{(1-success_a)*100:.2f}%", ha='center', fontsize=12, 
                      color='white' if (1-success_a) > 0.3 else 'black')
                
                # Circuit B
                ax2.bar(['Success', 'Failure'], [success_b, 1-success_b], color=['blue', 'orange'])
                ax2.set_title(f'Circuit B - Success Rate: {success_b*100:.2f}%', fontsize=14)
                ax2.set_ylim(0, 1)
                ax2.text(0, success_b/2, f"{success_b*100:.2f}%", ha='center', fontsize=12, 
                      color='white' if success_b > 0.3 else 'black')
                ax2.text(1, (1-success_b)/2, f"{(1-success_b)*100:.2f}%", ha='center', fontsize=12, 
                      color='white' if (1-success_b) > 0.3 else 'black')
                
                plt.tight_layout()
                plt.show()
                
                # Display parameters and difference
                diff = success_b - success_a
                diff_percent = diff * 100
                
                print("Circuit A Parameters:")
                print(f"- Depth: {circuit_a['depth']}")
                print(f"- Size: {circuit_a['size']}")
                print(f"- Width: {circuit_a['width']}")
                print(f"- Payload: {circuit_a['payload']}")
                print(f"Success Rate: {success_a:.4f} ({success_a*100:.2f}%)")
                
                print("\nCircuit B Parameters:")
                print(f"- Depth: {circuit_b['depth']}")
                print(f"- Size: {circuit_b['size']}")
                print(f"- Width: {circuit_b['width']}")
                print(f"- Payload: {circuit_b['payload']}")
                print(f"Success Rate: {success_b:.4f} ({success_b*100:.2f}%)")
                
                print(f"\nDifference (B - A): {diff:.4f} ({diff_percent:.2f}%)")
                if diff > 0:
                    print("Circuit B has a higher predicted success rate.")
                elif diff < 0:
                    print("Circuit A has a higher predicted success rate.")
                else:
                    print("Both circuits have the same predicted success rate.")
        
        # Connect sliders to update function
        for slider in [depth_a, size_a, width_a, payload_a, depth_b, size_b, width_b, payload_b]:
            slider.observe(update_comparison, names='value')
        
        # Create UI layout
        ui = widgets.VBox([
            widgets.HTML(value="<h3>Circuit Configuration Comparison</h3>"),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML(value="<b>Circuit A</b>"),
                    depth_a, size_a, width_a, payload_a
                ]),
                widgets.VBox([
                    widgets.HTML(value="<b>Circuit B</b>"),
                    depth_b, size_b, width_b, payload_b
                ])
            ]),
            output
        ])
        
        # Run initial update
        update_comparison()
        
        # Display the UI
        display(ui)
    
    def interactive_optimal_finder(self):
        """Interactive tool to find optimal circuit parameters for a given payload size"""
        if not WIDGETS_AVAILABLE:
            print("Error: ipywidgets not available. Install with: pip install ipywidgets")
            return
        
        # Create output widget
        output = widgets.Output()
        
        # Create slider for payload size
        payload_slider = widgets.IntSlider(
            value=3, min=1, max=10, step=1,
            description='Payload Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )
        
        # Create dropdown for parameter to optimize
        param_dropdown = widgets.Dropdown(
            options=[
                ('Circuit Depth', 'circuit_depth'),
                ('Circuit Size', 'circuit_size'),
                ('Circuit Width', 'circuit_width')
            ],
            value='circuit_depth',
            description='Optimize:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        )
        
        # Button to trigger optimization
        optimize_button = widgets.Button(
            description='Find Optimal Parameters',
            button_style='primary',
            layout=widgets.Layout(width='50%')
        )
        
        # Function to find optimal parameters
        def find_optimal_params(b):
            with output:
                # Clear previous output
                output.clear_output(wait=True)
                
                payload_size = payload_slider.value
                param_to_optimize = param_dropdown.value
                
                print(f"Finding optimal parameters for payload size {payload_size}...")
                print(f"Optimizing for {param_dropdown.options[param_dropdown.index][0]}...")
                
                # Set reasonable ranges for parameters
                ranges = {
                    'circuit_depth': np.arange(5, 51, 1),
                    'circuit_size': np.arange(5, 101, 5),
                    'circuit_width': np.arange(max(2, payload_size), 21, 1)
                }
                
                # Set reasonable defaults based on payload size
                default_width = payload_size + 1
                default_size = payload_size * 5
                default_depth = 20
                
                # Parameter to vary
                param_values = ranges[param_to_optimize]
                success_rates = []
                
                # Calculate success rate for each parameter value
                for val in param_values:
                    params = {
                        'circuit_depth': default_depth,
                        'circuit_size': default_size,
                        'circuit_width': default_width,
                        'payload_size': payload_size
                    }
                    params[param_to_optimize] = val
                    
                    success_rate = self.predict_success_rate(
                        params['circuit_depth'],
                        params['circuit_size'],
                        params['circuit_width'],
                        params['payload_size']
                    )
                    success_rates.append(success_rate)
                
                # Find the optimal value
                optimal_idx = np.argmax(success_rates)
                optimal_value = param_values[optimal_idx]
                max_success_rate = success_rates[optimal_idx]
                
                # Calculate optimal parameters
                optimal_params = {
                    'circuit_depth': default_depth,
                    'circuit_size': default_size,
                    'circuit_width': default_width,
                    'payload_size': payload_size
                }
                optimal_params[param_to_optimize] = optimal_value
                
                # Display results
                print("\nOptimal Parameters Found:")
                print(f"- Circuit Depth: {optimal_params['circuit_depth']}")
                print(f"- Circuit Size: {optimal_params['circuit_size']}")
                print(f"- Circuit Width: {optimal_params['circuit_width']}")
                print(f"- Payload Size: {optimal_params['payload_size']}")
                print(f"\nPredicted Success Rate: {max_success_rate:.4f} ({max_success_rate*100:.2f}%)")
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot optimization curve
                ax1.plot(param_values, success_rates, 'o-', linewidth=2)
                ax1.axvline(x=optimal_value, color='r', linestyle='--', 
                         label=f'Optimal {param_to_optimize} = {optimal_value}')
                ax1.set_xlabel(param_dropdown.options[param_dropdown.index][0], fontsize=12)
                ax1.set_ylabel('Predicted Success Rate', fontsize=12)
                ax1.set_title(f'Success Rate vs {param_dropdown.options[param_dropdown.index][0]}', fontsize=14)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot optimal success rate
                ax2.bar(['Success', 'Failure'], [max_success_rate, 1-max_success_rate], color=['green', 'red'])
                ax2.set_title(f'Optimal Success Rate: {max_success_rate*100:.2f}%', fontsize=14)
                ax2.set_ylabel('Probability', fontsize=12)
                ax2.set_ylim(0, 1)
                
                # Add text labels
                ax2.text(0, max_success_rate/2, f"{max_success_rate*100:.2f}%", ha='center', fontsize=12, 
                      color='white' if max_success_rate > 0.3 else 'black')
                ax2.text(1, (1-max_success_rate)/2, f"{(1-max_success_rate)*100:.2f}%", ha='center', fontsize=12, 
                      color='white' if (1-max_success_rate) > 0.3 else 'black')
                
                plt.tight_layout()
                plt.show()
        
        # Connect button to function
        optimize_button.on_click(find_optimal_params)
        
        # Create UI layout
        ui = widgets.VBox([
            widgets.HTML(value="<h3>Optimal Circuit Parameter Finder</h3>"),
            widgets.VBox([
                payload_slider,
                param_dropdown,
                optimize_button
            ]),
            output
        ])
        
        # Display the UI
        display(ui) 