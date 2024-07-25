from qiskit import QuantumCircuit
import numpy as np

def quantum_algorithm(probabilities):
    """
    Create a quantum circuit with one qubit per probability slot. Each qubit is rotated around the Y-axis
    based on the given probabilities.

    Args:
    probabilities (list of float): A list of probabilities for each slot. The probabilities should be in percentage form (0-100).

    Returns:
    QuantumCircuit: The constructed quantum circuit with rotations and measurements.
    """
    # Number of qubits/slots needed
    num_slots = len(probabilities)
    
    # Initialize a quantum circuit with `num_slots` qubits and `num_slots` classical bits for measurement
    qc = QuantumCircuit(num_slots, num_slots)

    # Apply Y-axis rotations based on the provided probabilities
    for i, prob in enumerate(probabilities):
        # Calculate the rotation angle theta based on the probability
        theta = 2 * np.arcsin(np.sqrt(prob / 100.0))
        # theta = 2 * math.acos(math.sqrt(prob))

        qc.ry(theta, i)  # Apply the rotation ry(theta) to the i-th qubit

    # Print the range of qubits being measured
    print(range(num_slots))
    
    # Measure all qubits and store the results in the corresponding classical bits
    qc.measure(range(num_slots), range(num_slots))

    return qc
