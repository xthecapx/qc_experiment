from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qbraid.providers import QbraidProvider
from enum import Enum

class QbraidDevice(Enum):
    SV1 = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    QIR = "qbraid_qir_simulator"
    TN1 = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"
    DM1 = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"
    AQUILA = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

class GateFidelity:
    def __init__(self):
        self.qubit = QuantumRegister(1, "Q")
        self.ebit0 = QuantumRegister(1, "A")
        self.ebit1 = QuantumRegister(1, "B")
        self.payload_size = 1
        self.aux = QuantumRegister(self.payload_size, "R")
        
    def create_protocol(self):
        protocol = QuantumCircuit(self.qubit, self.ebit0, self.ebit1)
        
        # Prepare ebit used for teleportation
        protocol.h(self.ebit0)
        protocol.cx(self.ebit0, self.ebit1)
        protocol.barrier()
        
        # Alice's operations
        protocol.cx(self.qubit, self.ebit0)
        protocol.h(self.qubit)
        protocol.barrier()
        
        # Alice measures and sends classical bits to Bob
        protocol.cx(self.ebit0, self.ebit1)
        protocol.cz(self.qubit, self.ebit1)
        protocol.barrier()
        
        return protocol
    
    def run_simulation(self, pre_gate='s', post_gate='sdg'):
        # Create test circuit
        test = QuantumCircuit(self.aux, self.qubit, self.ebit0, self.ebit1)
        
        # Entangle Q with R
        for qreg in self.aux:
            test.h(qreg)
            test.cx(qreg, self.qubit)
            # Configurable pre-gate
            getattr(test, pre_gate)(qreg)
        test.barrier()
        
        # Append protocol
        protocol = self.create_protocol()
        test = test.compose(protocol, qubits=range(self.payload_size, self.payload_size+3))
        test.barrier()
        
        # Verification steps
        for qreg in reversed(self.aux):
            # Configurable post-gate
            getattr(test, post_gate)(qreg)
            test.cx(qreg, self.ebit1)
            test.h(qreg)
        
        test.barrier()
        
        # Measurement
        result = ClassicalRegister(self.payload_size, "Test result")
        test.add_register(result)
        test.measure(self.aux, result)
        
        # Run simulation
        counts = AerSimulator().run(test).result().get_counts()
        return counts, test
    
    def run_qbraid(self, device: QbraidDevice = QbraidDevice.QIR, pre_gate='s', post_gate='sdg'):
        # Create test circuit
        test = QuantumCircuit(self.aux, self.qubit, self.ebit0, self.ebit1)
        
        # Entangle Q with R
        for qreg in self.aux:
            test.h(qreg)
            test.cx(qreg, self.qubit)
            getattr(test, pre_gate)(qreg)
        test.barrier()
        
        # Append protocol
        protocol = self.create_protocol()
        test = test.compose(protocol, qubits=range(self.payload_size, self.payload_size+3))
        test.barrier()
        
        # Verification steps
        for qreg in reversed(self.aux):
            getattr(test, post_gate)(qreg)
            test.cx(qreg, self.ebit1)
            test.h(qreg)
        
        test.barrier()
        
        # Measurement
        result = ClassicalRegister(self.payload_size, "Test result")
        test.add_register(result)
        test.measure(self.aux, result)
        
        # Run on qBraid
        provider = QbraidProvider()
        qbraid_device = provider.get_device(device.value)
        job = qbraid_device.run(test)
        counts = job.result().data.get_counts()
        
        return counts, test

# Create instance
fidelity_test = GateFidelity()

# Run with default gates (s/sdg)
counts, circuit = fidelity_test.run_simulation()

# Run with different gates
counts, circuit = fidelity_test.run_simulation(pre_gate='t', post_gate='tdg')

# Run with qBraid
counts, circuit = fidelity_test.run_qbraid()

# Visualize results
plot_histogram(counts)
circuit.draw(output='mpl')