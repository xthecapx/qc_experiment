import numpy as np
import random

class Payload:
    def __init__(self):
        self.gates = []
        self.gate_types = ['u', 'x', 'y', 'z']

    def generate_random_u_params(self):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        lambda_ = np.random.uniform(0, 2 * np.pi)

        return theta, phi, lambda_
    
    def add_random_gates(self, qc, num_gates):
        for _ in range(num_gates):
            self.add_random_gate(qc)

        return qc

    def add_random_gate(self, qc):
        random_gate = random.choice(self.gate_types)
        self.add_gate(qc, random_gate)

        return qc

    def add_gate(self, qc, gate_type):
        if gate_type == 'u':
            theta, phi, lambda_ = self.generate_random_u_params()
            qc.u(theta, phi, lambda_, 0)
            self.gates.append(('u', theta, phi, lambda_))
        elif gate_type == 'x':
            qc.x(0)
            self.gates.append(('x',))
        elif gate_type == 'y':
            qc.y(0)
            self.gates.append(('y',))
        elif gate_type == 'z':
            qc.z(0)
            self.gates.append(('z',))
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")
        
        return qc

    def apply_conjugate(self, qc):
        for gate in reversed(self.gates):
            if gate[0] == 'u':
                qc.u(-gate[1], -gate[3], -gate[2], 2)
            elif gate[0] == 'x':
                qc.x(2)
            elif gate[0] == 'y':
                qc.y(2)
            elif gate[0] == 'z':
                qc.z(2)

        return qc
