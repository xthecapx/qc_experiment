from qiskit import QuantumRegister, QuantumCircuit
from .oracle import Oracle

class Diffuser:
    def __init__(self, n_qubits: int, qubits: QuantumRegister, name: str = "Diffuser"):
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.name = name
        self.circuit = None
    
    def build(self) -> QuantumCircuit:
        diffuser = QuantumCircuit(self.qubits, name=self.name)
        
        diffuser.h(self.qubits)
        oracle = Oracle(self.n_qubits, self.qubits)
        diffuser.append(oracle.build({"0" * self.n_qubits}), list(range(self.n_qubits)))
        diffuser.h(self.qubits)

        self.circuit = diffuser
        return diffuser 