from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.result import marginal_counts
import numpy as np
from app.quantum.payload import Payload
import random

class QuantumGate:
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or []

    def apply(self, qc, qubit):
        if self.name == 'u':
            qc.u(*self.params, qubit)
        else:
            getattr(qc, self.name)(qubit)

    def apply_conjugate(self, qc, qubit):
        if self.name == 'u':
            qc.u(-self.params[0], -self.params[2], -self.params[1], qubit)
        elif self.name == 's':
            qc.sdg(qubit)
        elif self.name == 't':
            qc.tdg(qubit)
        else:
            # x and y are self-inverse
            self.apply(qc, qubit)

class TeleportationProtocol:
    def __init__(self):
        self.message_qubit = QuantumRegister(1, "M")
        self.alice_entangled = QuantumRegister(1, "A")
        self.bob_entangled = QuantumRegister(1, "B")
        self.circuit = QuantumCircuit(
            self.message_qubit, 
            self.alice_entangled, 
            self.bob_entangled
        )
        self._create_protocol()

    def _create_protocol(self):
        # Prepare the entangled pair (Bell state) between Alice and Bob
        self.circuit.h(self.alice_entangled)
        self.circuit.cx(self.alice_entangled, self.bob_entangled)
        self.circuit.barrier()

        # Alice's operations on her qubits
        self.circuit.cx(self.message_qubit, self.alice_entangled)
        self.circuit.h(self.message_qubit)
        self.circuit.barrier()

        # Bell measurement and classical communication
        self.circuit.cx(self.alice_entangled, self.bob_entangled)
        self.circuit.cz(self.message_qubit, self.bob_entangled)
        self.circuit.barrier()

    def draw(self):
        return self.circuit.draw(output='mpl')

class TeleportationValidator:
    def __init__(self, payload_size: int = 3, num_gates: int = 1):
        self.gates = {}
        self.payload_size = payload_size
        self.num_gates = num_gates
        self.gate_types = {
            'u': lambda: QuantumGate('u', self._generate_random_u_params()),
            'x': lambda: QuantumGate('x'),
            'y': lambda: QuantumGate('y'),
            's': lambda: QuantumGate('s'),
            't': lambda: QuantumGate('t')
        }
        self.auxiliary_qubits = QuantumRegister(payload_size, "R")
        self.protocol = TeleportationProtocol()
        self.result = ClassicalRegister(payload_size, "Test result")
        self.payload = Payload(payload_size)
        self.circuit = self._create_test_circuit()

    def _generate_random_u_params(self):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        lambda_ = np.random.uniform(0, 2 * np.pi)

        return theta, phi, lambda_

    def _add_random_gate(self, qc: QuantumCircuit, qubit: QuantumRegister):
        gate_type = random.choice(list(self.gate_types.keys()))
        gate = self.gate_types[gate_type]()
        
        # If this qubit already has gates, append to the list
        if qubit in self.gates:
            if isinstance(self.gates[qubit], list):
                self.gates[qubit].append(gate)
            else:
                self.gates[qubit] = [self.gates[qubit], gate]
        else:
            self.gates[qubit] = gate
            
        gate.apply(qc, qubit)

    def _create_test_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(
            self.auxiliary_qubits, 
            self.protocol.message_qubit,
            self.protocol.alice_entangled, 
            self.protocol.bob_entangled
        )

        self._create_payload(circuit)
        circuit.barrier()

        circuit = circuit.compose(
            self.protocol.circuit, 
            qubits=range(self.payload_size, self.payload_size + 3)
        )
        circuit.barrier()

        self._create_validation(circuit)
        circuit.barrier()

        circuit.add_register(self.result)
        circuit.measure(self.auxiliary_qubits, self.result)

        return circuit

    def _create_payload(self, circuit: QuantumCircuit):
        # Apply initial operations to all qubits
        for qubit in self.auxiliary_qubits:
            circuit.h(qubit)
            circuit.cx(qubit, self.protocol.message_qubit)
        
        # Ensure each qubit gets at least one gate
        gates_per_qubit = self.num_gates // self.payload_size
        remaining_gates = self.num_gates % self.payload_size
        
        # First, distribute gates evenly
        for qubit in self.auxiliary_qubits:
            for _ in range(gates_per_qubit):
                self._add_random_gate(circuit, qubit)
        
        # Then distribute remaining gates randomly
        if remaining_gates:
            selected_qubits = random.sample(list(self.auxiliary_qubits), remaining_gates)
            for qubit in selected_qubits:
                self._add_random_gate(circuit, qubit)

    def _create_validation(self, circuit: QuantumCircuit):
        for qubit in reversed(self.auxiliary_qubits):
            # Get gates for this qubit if any were applied
            if qubit in self.gates:
                gates = self.gates[qubit]
                if isinstance(gates, list):
                    # Apply conjugates of all gates in reverse order
                    for gate in reversed(gates):
                        gate.apply_conjugate(circuit, qubit)
                else:
                    # Single gate case
                    gates.apply_conjugate(circuit, qubit)
            
            # Apply the inverse of the initial operations
            circuit.cx(qubit, self.protocol.bob_entangled)
            circuit.h(qubit)

    def draw(self):
        return self.circuit.draw(output='mpl')

    def simulate(self):
        simulator = AerSimulator()
        result = simulator.run(self.circuit).result()
        counts = result.get_counts()
        return counts

    def plot_results(self):
        counts = self.simulate()
        return plot_histogram(counts)

    def get_metrics(self):
        # Get simulation results
        counts = self.simulate()
        
        # Results-based metrics
        results_metrics = {
            "counts": counts,
            "success_rate": counts.get('0' * self.payload_size, 0) / sum(counts.values()),
        }
        
        # Circuit metrics from Qiskit
        circuit_metrics = {
            "depth": self.circuit.depth(),
            "width": self.circuit.width(),  # Added width
            "size": self.circuit.size(),
            "count_ops": self.circuit.count_ops(),  # Added operation count by type
        }
        
        # Configuration metrics
        config_metrics = {
            "payload_size": self.payload_size,
            "num_gates": self.num_gates,
        }
        
        # Custom gate distribution from our tracking
        gate_distribution = {}
        for qubit_gates in self.gates.values():
            if isinstance(qubit_gates, list):
                for gate in qubit_gates:
                    gate_distribution[gate.name] = gate_distribution.get(gate.name, 0) + 1
            else:
                gate_distribution[gate.name] = gate_distribution.get(gate.name, 0) + 1
        
        return {
            "results_metrics": results_metrics,
            "circuit_metrics": circuit_metrics,
            "config_metrics": config_metrics,
            "custom_gate_distribution": gate_distribution
        }


# Usage example:
if __name__ == "__main__":
    # Create and visualize just the teleportation protocol
    # protocol = TeleportationProtocol()
    # protocol.draw()

    # Create and run the validation
    validator = TeleportationValidator(payload_size=5, num_gates=100)
    validator.draw()
    validator.plot_results()