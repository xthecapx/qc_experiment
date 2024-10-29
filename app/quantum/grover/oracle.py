from qiskit import QuantumRegister, QuantumCircuit

class Oracle:
    def __init__(self, n_qubits: int, qubits: QuantumRegister, name: str = "Oracle"):
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.name = name
        self.circuit = None
    
    def _flip(self, target: str, circuit: QuantumCircuit, qubit: str = "0") -> None:
        for i in range(len(target)):
            if target[i] == qubit:
                circuit.x(i)  # Pauli-X gate
    
    def build(self, targets: set[str]) -> QuantumCircuit:
        oracle_circuit = QuantumCircuit(self.qubits, name=self.name)

        for target in targets:
            target = target[::-1]
            self._flip(target, oracle_circuit, "0")

            oracle_circuit.h(self.n_qubits - 1)
            oracle_circuit.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
            oracle_circuit.h(self.n_qubits - 1)

            self._flip(target, oracle_circuit, "0")

        self.circuit = oracle_circuit
        return oracle_circuit 