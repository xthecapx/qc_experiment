from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from .oracle_memory import OracleMemory
from .diffuser_memory import DiffuserMemory
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import transpile
from math import pi, sqrt
from heapq import nlargest
from typing import TypedDict, List, Optional

class SimulationResult(TypedDict):
    found_states: str  # Contains formatted string with binary and decimal found states
    targets: str  # Contains formatted string with binary and decimal targets
    error: Optional[str]  # Present only when targets are not found
    targets_accuracy: Optional[List[str]]  # Present only when targets are found
    metrics: dict  # Circuit metrics like depth, size, etc.

class GroverMemory():
    def __init__(self, 
                 n_qubits: int,
                 optimization_level: int = 2,
                 shots: int = 10000,
                 name: str = "Grover Memory"):
        """Initialize GroverMemory with configuration.
        
        Args:
            n_qubits (int): Number of qubits to use for memory
            optimization_level (int, optional): Circuit optimization level. Defaults to 2.
            shots (int, optional): Number of simulation shots. Defaults to 10000.
            name (str, optional): Circuit name. Defaults to "Grover Memory".
        """
        self.optimization_level = optimization_level
        self.shots = shots
        self.name = name
        self.n_qubits = n_qubits
        self.max_storable = 2**n_qubits - 1  # Maximum value that can be stored

        # Components start as None
        self.oracle = None
        self.diffuser = None
        self._circuit = None
        self._qubits = None
        self.density_matrix = None
        self.read_circuit = None
        self.targets = set()

        self._initialize_circuit()  # Call initialize once during init

    def _initialize_circuit(self) -> None:
        """Private method to initialize the circuit once."""
        self._qubits = QuantumRegister(self.n_qubits, "qubit")
        self._circuit = QuantumCircuit(self._qubits, name=self.name)
        self._circuit.h(self._qubits)
        self._circuit.barrier()

    @property
    def circuit(self) -> QuantumCircuit:
        """Getter for quantum circuit that always returns a copy."""
        return self._circuit.copy()

    @circuit.setter
    def circuit(self, new_circuit: QuantumCircuit) -> None:
        """Setter for quantum circuit."""
        self._circuit = new_circuit

    def _simulate_circuit(self) -> dict:
        """Simulate the quantum circuit and return measurement results."""
        backend = AerSimulator(method="density_matrix")
        transpiled = transpile(self.read_circuit, backend, optimization_level=self.optimization_level)
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
            'targets': f"TARGET(S):\nBinary = {self.targets}\nDecimal = {[int(key, 2) for key in self.targets]}\n",
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

    def memory(self, values: set[int], operation: str = "write"):
        """Perform memory operations (write/delete).
        
        Args:
            values (set[int]): Values to write to or delete from memory
            operation (str): Operation to perform ("write" or "delete")
        """
        circuit = self.circuit
        binary_values = {f"{s:0{self.n_qubits}b}" for s in values}
        
        if operation == "write":
            self.targets.update(binary_values)
        elif operation == "delete":
            self.targets.difference_update(binary_values)
        else:
            raise ValueError("Operation must be either 'write' or 'delete'")

        # Only proceed if there are targets in memory
        if self.targets:
            self.oracle = OracleMemory(n_qubits=self.n_qubits, qubits=self._qubits)
            self.diffuser = DiffuserMemory(n_qubits=self.n_qubits, qubits=self._qubits, oracle_memory=self.oracle)
            iterations = int((pi / 4) * sqrt((2 ** self.n_qubits) / len(self.targets)))

            for _ in range(iterations):
                circuit.append(self.oracle.build(self.targets), list(range(self.n_qubits)))
                circuit.append(self.diffuser.build(), list(range(self.n_qubits)))

            self.density_matrix = DensityMatrix(circuit)
            circuit.measure_all()
            self.read_circuit = circuit
        else:
            # Reset components if no targets remain
            self.oracle = None
            self.diffuser = None
            self.density_matrix = None
            self.read_circuit = None

    def write(self, values: set[int]) -> dict:
        """Write values into quantum memory.
        
        Args:
            values (set[int]): Values to store in memory
            
        Returns:
            dict: Status of the write operation
        """
        # Validate values
        for value in values:
            if value > self.max_storable:
                return {
                    "success": False,
                    "error": f"Value {value} is too big for this memory. Maximum value allowed: {self.max_storable}"
                }
            
        self.memory(values, operation="write")
        
        return {
            "success": True,
            "message": f"Successfully wrote values {values} to quantum memory"
        }

    def delete(self, values: set[int]) -> dict:
        """Delete values from quantum memory.
        
        Args:
            values (set[int]): Values to delete from memory
            
        Returns:
            dict: Status of the delete operation
        """
        # Convert values to binary format for checking
        binary_values = {f"{value:0{self.n_qubits}b}" for value in values}
        
        # Check if all values exist in memory
        missing_values = binary_values - self.targets
        if missing_values:
            missing_decimal = [int(value, 2) for value in missing_values]
            return {
                "success": False,
                "error": f"Values {missing_decimal} are not present in memory"
            }
            
        self.memory(values, operation="delete")
        
        return {
            "success": True,
            "message": f"Successfully deleted values {values} from quantum memory"
        }

    def read(self) -> dict:
        """Read all values stored in quantum memory through simulation."""
        # Check if memory is empty
        if not self.targets or self.oracle is None or self.diffuser is None:
            return {
                "success": True,
                "message": "Memory is empty",
                "stored_values": set(),
                "measurements": {}
            }
        
        return self._simulate_circuit()
