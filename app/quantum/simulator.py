from qiskit_aer import AerSimulator
from app.quantum.circuit import quantum_algorithm

def setup_simulator(probabilities):
    """
    Set up and run a quantum circuit based on the given probabilities. It returns the most likely outcome,
    the quantum circuit, and the count of all outcomes.

    Args:
    probabilities (list of float): A list of probabilities for each slot. The probabilities should be in percentage form (0-100).

    Returns:
    tuple: The most likely outcome (as a bitstring), the quantum circuit, and the counts of all outcomes.
    """
    # Generate the quantum circuit using the provided probabilities
    qc = quantum_algorithm(probabilities)
    
    # Display the circuit diagram
    # display(qc.draw(output="mpl"))
    
    # Use Aer's qasm_simulator to simulate the circuit
    backend = AerSimulator()

    # Run the circuit on the simulator with 100 shots
    job = backend.run(qc, shots = 1000)
    
    # Grab results from the job
    result = job.result()
    counts = result.get_counts()
    
    # Print the outcome counts
    print(counts)
    
    # Determine the most likely outcome
    outcome = max(counts, key=counts.get)
    
    # Return the most likely outcome, the quantum circuit, and the counts of outcomes
    return outcome, qc, counts