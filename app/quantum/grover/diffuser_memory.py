from qiskit import QuantumRegister, QuantumCircuit
from .oracle_memory import OracleMemory

class DiffuserMemory:
    def __init__(self, n_qubits: int, qubits: QuantumRegister, oracle_memory: OracleMemory, name: str = "Diffuser Memory"):
        """Initialize Diffuser Memory.
        
        Args:
            n_qubits (int): Number of qubits
            qubits (QuantumRegister): Quantum register to use
            oracle_memory (OracleMemory): Instance of OracleMemory to use for consistency
            name (str, optional): Name of the diffuser. Defaults to "Diffuser Memory".
        """
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.name = name
        self._circuit = None
        self._oracle_memory = oracle_memory
    
    def build(self) -> QuantumCircuit:
        """Build diffuser circuit using the shared oracle memory instance.
        
        Returns:
            QuantumCircuit: Diffuser circuit
        """
        diffuser = QuantumCircuit(self.qubits, name=self.name)
        
        # Apply H gates to all qubits
        diffuser.h(self.qubits)
        
        # Use the shared oracle memory instance to mark the zero state
        diffuser.append(
            self._oracle_memory.build(targets={"0" * self.n_qubits}), 
            list(range(self.n_qubits))
        )
        
        # Apply H gates to all qubits again
        diffuser.h(self.qubits)

        self._circuit = diffuser

        return diffuser 