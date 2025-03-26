def plot_experiment_results(experiment_results):
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert string representation to list if needed
    if isinstance(experiment_results, str):
        import ast

        experiment_results = ast.literal_eval(experiment_results)

    # Handle the case where experiment_results is a list of result groups
    if isinstance(experiment_results, list) and "experiments" in experiment_results[0]:
        # Flatten the experiments from all groups
        all_experiments = []
        for group in experiment_results:
            all_experiments.extend(group["experiments"])
        experiment_results = all_experiments

    # Organize data by payload size and num_gates
    data = {}
    for exp in experiment_results:
        if exp["status"] != "completed":
            continue

        payload = exp["experiment_params"]["payload_size"]
        gates = exp["experiment_params"]["num_gates"]
        success = exp["results_metrics"]["success_rate"] * 100

        if payload not in data:
            data[payload] = {"gates": [], "success": []}
        data[payload]["gates"].append(gates)
        data[payload]["success"].append(success)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot lines for each payload size with different colors and markers
    colors = ["b", "g", "r", "c", "m"]
    for i, (payload, values) in enumerate(sorted(data.items())):
        # Sort the data points by number of gates
        points = sorted(zip(values["gates"], values["success"]))
        gates, success = zip(*points)

        plt.plot(
            gates,
            success,
            marker="o",
            color=colors[i % len(colors)],
            label=f"Payload Size {payload}",
        )

    plt.xlabel("Number of Gates")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rates by Payload Size and Number of Gates")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

