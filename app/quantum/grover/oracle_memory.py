from qiskit import QuantumRegister, QuantumCircuit
from typing import Optional

class OracleMemory:
    def __init__(self, n_qubits: int, qubits: QuantumRegister, name: str = "Oracle Memory"):
        """Initialize Oracle Memory.
        
        Args:
            n_qubits (int): Number of qubits
            qubits (QuantumRegister): Quantum register to use
            name (str, optional): Name of the oracle. Defaults to "Oracle Memory".
        """
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.name = name
        self._circuit: Optional[QuantumCircuit] = None
    
    def _flip(self, target: str, circuit: QuantumCircuit, qubit: str = "0") -> None:
        """Apply X gates to flip qubits based on target string.
        
        Args:
            target (str): Target binary string
            circuit (QuantumCircuit): Circuit to apply flips to
            qubit (str, optional): Qubit value to flip on. Defaults to "0".
        """
        for i in range(len(target)):
            if target[i] == qubit:
                circuit.x(i)
    
    def build(self, targets: set[str]) -> QuantumCircuit:
        """Build or return existing oracle circuit for given targets.
        
        Args:
            targets (Set[str]): Target states to mark
            
        Returns:
            QuantumCircuit: Oracle circuit
        """
        oracle_circuit = QuantumCircuit(self.qubits, name=self.name)

        for target in targets:
            target = target[::-1]  # Reverse for correct qubit ordering
            self._flip(target, oracle_circuit, "0")

            # Apply phase flip
            oracle_circuit.h(self.n_qubits - 1)
            oracle_circuit.mcx(list(range(self.n_qubits - 1)), self.n_qubits - 1)
            oracle_circuit.h(self.n_qubits - 1)

            # Uncompute flips
            self._flip(target, oracle_circuit, "0")

        self._circuit = oracle_circuit
        
        return oracle_circuit
