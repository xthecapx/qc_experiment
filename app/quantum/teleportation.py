from qiskit import QuantumCircuit, transpile
from app.quantum.payload import Payload
from qiskit_aer import AerSimulator
from qiskit.result import marginal_counts
# from qiskit.visualization import plot_histogram, plot_state_city
import numpy as np
from qbraid.transpiler import transpile as qbraid_transpile
from qbraid.runtime import QbraidProvider

def generate_teleportation_circuit():
    # Create teleportation circuit
    qc = QuantumCircuit(3, 3)
    
    # Apply random U gate to the first qubit
    payload = Payload()
    qc = payload.add_random_gate(qc)
    
    # Create Bell pair
    qc.h(1)
    qc.cx(1, 2)
    
    # Teleportation protocol
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.cx(1, 2)
    qc.cz(0, 2)
    
    # Apply inverse U gate
    qc = payload.apply_conjugate(qc)
    
    # Measure the final state
    qc.measure(2, 2)

    return qc, payload

def teleportation_experiment(N):
    qc, payload = generate_teleportation_circuit()

    # display(qc.draw())
    simulator = AerSimulator()
    job = simulator.run(qc, shots=N)
    result = job.result()
    
    # Analyze results
    counts = marginal_counts(result.get_counts(), [2])
    success_rate = counts.get('0', 0) / N

    counts = result.get_counts(qc)
    # display(plot_histogram(counts, title='Bell-State counts'))

    return success_rate, counts, payload.gates

def qbraid_teleportation_experiment(N):
    # Generate the Qiskit circuit
    # calcular la cantidad de qubits, y ver como cambian los resultados al aumentar la complejidad de las compuertas
    qiskit_circuit = generate_teleportation_circuit()
    
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

    jobs = device.run(qiskit_circuit, shots=N)
    results = [job.result() for job in jobs]
    
    # Analyze results
    counts = results[0].raw_counts()
    print(counts)
    success_rate = 100
    
    return success_rate, counts

