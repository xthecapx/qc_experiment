from qiskit import QuantumCircuit, transpile
from app.quantum.payload import Payload
from qiskit_aer import AerSimulator
from qiskit.result import marginal_counts
# from qiskit.visualization import plot_histogram, plot_state_city
import numpy as np
from qbraid.transpiler import transpile as qbraid_transpile
from qbraid.runtime import QbraidProvider

def generate_teleportation_circuit(num_gates, num_payload_qubits):
    total_qubits = num_payload_qubits + 3
    # Create teleportation circuit
    qc = QuantumCircuit(total_qubits, 3)

    print(num_payload_qubits, 'num_payload_qubits', total_qubits, 'total_qubits')
    
    # Apply random U gate to the first qubit
    payload = Payload(num_payload_qubits)
    qc = payload.add_random_gates(qc, num_gates)
    
    # Create Bell pair
    qc.h(num_payload_qubits + 1)
    qc.cx(num_payload_qubits + 1, num_payload_qubits + 2)
    
    # Teleportation protocol
    qc.cx(num_payload_qubits, num_payload_qubits + 1)
    qc.h(num_payload_qubits)
    qc.measure(num_payload_qubits, 0)
    qc.measure(num_payload_qubits + 1, 1)
    qc.cx(num_payload_qubits + 1, num_payload_qubits + 2)
    qc.cz(num_payload_qubits, num_payload_qubits + 2)
    
    # # Apply inverse U gate
    # inverse_gate = UnitaryGate(payload.inverse_unitary)
    # qc = qc.append(inverse_gate, [num_payload_qubits + 2])
    qc = payload.apply_conjugate(qc)
    print('Input dimensions:', payload.inverse_unitary.input_dims())
    print('Output dimensions:', payload.inverse_unitary.output_dims())
    
    # # Measure the final state
    qc.measure(num_payload_qubits + 2, 2)

    return qc, payload

def teleportation_experiment(shots, num_gates, num_payload_qubits):
    qc, payload = generate_teleportation_circuit(num_gates, num_payload_qubits)

    # display(qc.draw())
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    
    # Analyze results
    counts = marginal_counts(result.get_counts(), [2])
    success_rate = counts.get('0', 0) / shots

    counts = result.get_counts(qc)
    # display(plot_histogram(counts, title='Bell-State counts'))

    return success_rate, counts, payload.gates, qc.depth()

def qbraid_teleportation_experiment(N):
    # Generate the Qiskit circuit
    qc, payload = generate_teleportation_circuit(2, 0)

    # Set up qBraid provider and backend
    # print(api_key)
    # print(qbraid_circuit)
    provider = QbraidProvider()
    devices = provider.get_devices()
    print(devices)

    device = provider.get_device("qbraid_qir_simulator")
    print(device.metadata())
    
    # Run the simulation
    # run_input = [qiskit_circuit]

    job = device.run(qc, shots=N)
    result = job.result()
    
    # Analyze results
    counts = result.data.get_counts()
    print(counts)
    success_rate = 100
    
    return success_rate, counts, payload.gates, qc.depth()

