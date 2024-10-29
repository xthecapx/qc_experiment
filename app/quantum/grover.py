from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from math import pi, sqrt, log2, ceil
from heapq import nlargest
from qiskit import transpile
from qiskit_aer import AerSimulator
from typing import TypedDict, List, Optional

class Oracle:
    def __init__(self, n_qubits: int, qubits: QuantumRegister, name: str = "Oracle"):
        """Initialize Oracle with configuration.
        
        Args:
            n_qubits (int): Number of qubits
            qubits (QuantumRegister): Quantum register
            name (str, optional): Circuit name. Defaults to "Oracle".
        """
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.name = name
        self.circuit = None
    
    def _flip(self, target: str, circuit: QuantumCircuit, qubit: str = "0") -> None:
        """Flips qubit in target state.

        Args:
            target (str): Binary string representing target state.
            circuit (QuantumCircuit): Quantum circuit.
            qubit (str, optional): Qubit to flip. Defaults to "0".
        """
        for i in range(len(target)):
            if target[i] == qubit:
                circuit.x(i)  # Pauli-X gate
    
    def build(self, targets: set[str]) -> QuantumCircuit:
        """Mark target state(s) with negative phase."""
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

class Diffuser:
    def __init__(self, n_qubits: int, qubits: QuantumRegister, name: str = "Diffuser"):
        """Initialize Diffuser with configuration.
        
        Args:
            n_qubits (int): Number of qubits
            qubits (QuantumRegister): Quantum register
            name (str, optional): Circuit name. Defaults to "Diffuser".
        """
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.name = name
        self.circuit = None
    
    def build(self) -> QuantumCircuit:
        """Build the diffuser circuit."""
        diffuser = QuantumCircuit(self.qubits, name=self.name)
        
        diffuser.h(self.qubits)
        oracle = Oracle(self.n_qubits, self.qubits)
        diffuser.append(oracle.build({"0" * self.n_qubits}), list(range(self.n_qubits)))
        diffuser.h(self.qubits)

        self.circuit = diffuser
        
        return diffuser

class SimulationResult(TypedDict):
    found_states: str  # Contains formatted string with binary and decimal found states
    targets: str  # Contains formatted string with binary and decimal targets
    error: Optional[str]  # Present only when targets are not found
    targets_accuracy: Optional[List[str]]  # Present only when targets are found
    metrics: dict  # Circuit metrics like depth, size, etc.

class Grover:
    def __init__(self, 
                 search_values: set[int] = {1, 2, 3},
                 optimization_level: int = 2,
                 shots: int = 10000,
                 name: str = "Grover Circuit"):
        """Initialize Grover with configuration.
        
        Args:
            search_values (set[int], optional): Values to search for. Defaults to {1, 2, 3}.
            optimization_level (int, optional): Circuit optimization level. Defaults to 2.
            shots (int, optional): Number of simulation shots. Defaults to 10000.
            name (str, optional): Circuit name. Defaults to "Grover Circuit".
        """
        self.search_values = search_values
        self.optimization_level = optimization_level
        self.shots = shots
        self.name = name
        
        # Calculate required number of qubits based on largest search value
        max_value = max(search_values)
        self.n_qubits = ceil(log2(max_value + 1))  # +1 because we need to represent 0 to max_value
        
        # Derived attributes
        self.qubits = QuantumRegister(self.n_qubits, "qubit")
        self.targets = {f"{s:0{self.n_qubits}b}" for s in self.search_values}
        
        # Initialize components
        self.oracle = Oracle(self.n_qubits, self.qubits)
        self.diffuser = Diffuser(self.n_qubits, self.qubits)
        
        # Build the circuit
        self.circuit, self.density_matrix = self.build()
    
    def build(self) -> tuple[QuantumCircuit, DensityMatrix]:
        """Create quantum circuit for Grover's algorithm."""
        grover = QuantumCircuit(self.qubits, name=self.name)

        grover.h(self.qubits)
        grover.barrier()
        
        iterations = int((pi / 4) * sqrt((2 ** self.n_qubits) / len(self.targets)))
        for _ in range(iterations):
            grover.append(self.oracle.build(self.targets), list(range(self.n_qubits)))
            grover.append(self.diffuser.build(), list(range(self.n_qubits)))

        density_matrix = DensityMatrix(grover)
        grover.measure_all()
        
        return grover, density_matrix

    def simulate(self) -> SimulationResult:
        """Execute the simulation and return the results.

        Returns:
            SimulationResult: A dictionary containing:
                - found_states: Formatted string showing binary and decimal representations of found states
                - targets: Formatted string showing binary and decimal representations of targets
                - error: Error message if targets are not found (optional)
                - targets_accuracy: List containing accuracy percentage if targets are found (optional)
                - metrics: Dictionary containing circuit metrics:
                    - depth: Circuit depth
                    - width: Number of qubits
                    - size: Total number of operations
                    - operations: Count of each operation type
        """
        backend = AerSimulator(method="density_matrix")
        transpiled = transpile(self.circuit, backend, optimization_level=self.optimization_level)
        simulation = backend.run(transpiled, shots=self.shots)
        counts = simulation.result().get_counts()
        found_states = {state: counts.get(state) for state in nlargest(len(self.targets), counts, key=counts.get)}
        found_states = list(found_states.keys())

        # Collect circuit metrics
        metrics = {
            'depth': transpiled.depth(),
            'width': transpiled.width(),
            'size': transpiled.size(),
            'operations': transpiled.count_ops()
        }

        res: SimulationResult = {
            'found_states': f"FOUND STATE(S):\nBinary = {found_states}\nDecimal = {[int(key, 2) for key in found_states]}\n",
            'targets': f"TARGET(S):\nBinary = {self.targets}\nDecimal = {self.search_values}\n",
            'metrics': metrics
        }

        # Check if all targets were found
        missing_targets = self.targets - set(found_states)
        if missing_targets:
            missing_decimal = [int(target, 2) for target in missing_targets]
            res['error'] = f"Missing target(s): Binary={missing_targets}, Decimal={missing_decimal}"
            res['targets_accuracy'] = None
        else:
            # Calculate global accuracy (how often we found any target)
            total_shots = sum(counts.values())
            target_shots = sum(counts.get(target, 0) for target in self.targets)
            global_accuracy = target_shots / total_shots

            # Calculate distribution for each target
            target_distributions = []
            for target in self.targets:
                target_frequency = counts.get(target, 0)
                distribution = target_frequency / total_shots
                target_distributions.append(
                    f"Target {target} (dec: {int(target, 2)}) was found {distribution:.2%} of the times"
                )
            
            res['error'] = None
            res['targets_accuracy'] = [
                f"Global accuracy: {global_accuracy:.2%}",
                "Distribution of findings:",
                *target_distributions
            ]
        
        return res
