import numpy as np
from qiskit.quantum_info import Operator
import random

class Payload:
    def __init__(self, num_payload_qubits):
        self.unitary = None
        self.inverse_unitary = None
        self.num_payload_qubits = num_payload_qubits
        self.gates = []
        self.gate_types = ['u', 'x', 'y', 'z']

    def generate_random_u_params(self):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        lambda_ = np.random.uniform(0, 2 * np.pi)

        return theta, phi, lambda_
    
    def add_random_gates(self, qc, num_gates):
        if (self.num_payload_qubits > 0):
            self.entangle_payload(qc)

        for _ in range(num_gates):
            self.add_random_gate(qc)

        self.unitary = Operator.from_circuit(qc)
        self.inverse_unitary = Operator.from_circuit(qc.inverse())

        return qc

    def add_random_gate(self, qc):
        random_gate = random.choice(self.gate_types)
        self.add_gate(qc, random_gate)

        return qc

    def add_gate(self, qc, gate_type):
        payload_qbit = self.num_payload_qubits

        if gate_type == 'u':
            theta, phi, lambda_ = self.generate_random_u_params()
            qc.u(theta, phi, lambda_, payload_qbit)
            self.gates.append(('u', theta, phi, lambda_))
        elif gate_type == 'x':
            qc.x(payload_qbit)
            self.gates.append(('x',))
        elif gate_type == 'y':
            qc.y(payload_qbit)
            self.gates.append(('y',))
        elif gate_type == 'z':
            qc.z(payload_qbit)
            self.gates.append(('z',))
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")
        
        return qc

    def apply_conjugate(self, qc):
        measure_qbit = self.num_payload_qubits + 2

        for gate in reversed(self.gates):
            if gate[0] == 'u':
                qc.u(-gate[1], -gate[3], -gate[2], measure_qbit)
            elif gate[0] == 'x':
                qc.x(measure_qbit)
            elif gate[0] == 'y':
                qc.y(measure_qbit)
            elif gate[0] == 'z':
                qc.z(measure_qbit)

        return qc
    
    def entangle_payload(self, qc):
        for i in range(self.num_payload_qubits):
            qc.h(i)
        for i in range(self.num_payload_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.num_payload_qubits - 1, self.num_payload_qubits)
        return qc
