from teleportation_validator import Experiments
import matplotlib.pyplot as plt

# Create an instance of the Experiments class
experiments = Experiments()

# Run the fixed payload experiments with only 1 iteration per payload size for testing
# Set run_on_ibm to False to run in simulation mode
results = experiments.run_fixed_payload_experiments(
    iterations=1,  # Just 1 iteration for testing
    run_on_ibm=False,
    show_circuit=True,
    show_histogram=True
)

# Print summary of results
print("\nExperiment Results Summary:")
print(f"Total experiments: {len(results)}")
for payload_size in range(1, 6):
    payload_results = results[results['payload_size'] == payload_size]
    avg_success = payload_results['success_rate'].mean() * 100
    print(f"Payload size {payload_size}: {len(payload_results)} experiments, Average success rate: {avg_success:.2f}%")

# Plot success rates by payload size
plt.figure(figsize=(10, 6))
for payload_size in range(1, 6):
    payload_results = results[results['payload_size'] == payload_size]
    plt.bar(f"Payload {payload_size}", payload_results['success_rate'].mean() * 100, label=f"Gates: {payload_size*3}")

plt.xlabel('Payload Size')
plt.ylabel('Success Rate (%)')
plt.title('Average Success Rates by Payload Size')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show() 