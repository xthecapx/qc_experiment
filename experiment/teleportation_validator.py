from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import numpy as np
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
    def __init__(self, use_barriers: bool = True):
        self.message_qubit = QuantumRegister(1, "M")
        self.alice_entangled = QuantumRegister(1, "A")
        self.bob_entangled = QuantumRegister(1, "B")
        self.use_barriers = use_barriers
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
        if self.use_barriers:
            self.circuit.barrier()

        # Alice's operations on her qubits
        self.circuit.cx(self.message_qubit, self.alice_entangled)
        self.circuit.h(self.message_qubit)
        if self.use_barriers:
            self.circuit.barrier()

        # Bell measurement and classical communication
        self.circuit.cx(self.alice_entangled, self.bob_entangled)
        self.circuit.cz(self.message_qubit, self.bob_entangled)
        if self.use_barriers:
            self.circuit.barrier()

    def draw(self):
        return self.circuit.draw(output='mpl')

class TeleportationValidator:
    def __init__(self, payload_size: int = 3, gates: list | int = None, use_barriers: bool = True):
        self.gates = {}
        self.payload_size = payload_size
        self.use_barriers = use_barriers
        self.gate_types = {
            'u': lambda: QuantumGate('u', self._generate_random_u_params()),
            'x': lambda: QuantumGate('x'),
            'y': lambda: QuantumGate('y'),
            's': lambda: QuantumGate('s'),
            't': lambda: QuantumGate('t')
        }
        
        # Handle gates parameter
        if isinstance(gates, int):
            # Generate random gates if gates is a number
            available_gates = list(self.gate_types.keys())
            self.input_gates = [random.choice(available_gates) for _ in range(gates)]
        else:
            # Use provided gates list or empty list if None
            self.input_gates = gates or []
        
        self.auxiliary_qubits = QuantumRegister(payload_size, "R")
        self.protocol = TeleportationProtocol(use_barriers=use_barriers)
        self.result = ClassicalRegister(payload_size, "test_result")
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
        if self.use_barriers:
            circuit.barrier()
    
        circuit = circuit.compose(
            self.protocol.circuit, 
            qubits=range(self.payload_size, self.payload_size + 3)
        )
        if self.use_barriers:
            circuit.barrier()
    
        self._create_validation(circuit)
        if self.use_barriers:
            circuit.barrier()
    
        circuit.add_register(self.result)
        circuit.measure(self.auxiliary_qubits, self.result)
    
        return circuit

    def _create_payload(self, circuit: QuantumCircuit):
        # First apply initial operations to all qubits
        for qubit in self.auxiliary_qubits:
            circuit.h(qubit)
            circuit.cx(qubit, self.protocol.message_qubit)
        
        if self.input_gates:
            # Calculate gates per qubit
            gates_per_qubit = len(self.input_gates) // self.payload_size
            remaining_gates = len(self.input_gates) % self.payload_size
            
            # Distribute gates across qubits
            gate_index = 0
            for i, qubit in enumerate(self.auxiliary_qubits):
                # Calculate how many gates this qubit should get
                num_gates = gates_per_qubit + (1 if i < remaining_gates else 0)
                
                # Apply the gates assigned to this qubit
                qubit_gates = []
                for _ in range(num_gates):
                    if gate_index < len(self.input_gates):
                        gate_name = self.input_gates[gate_index]
                        if gate_name in self.gate_types:
                            gate = self.gate_types[gate_name]()
                            qubit_gates.append(gate)
                            gate.apply(circuit, qubit)
                        gate_index += 1
                
                if qubit_gates:
                    self.gates[qubit] = qubit_gates if len(qubit_gates) > 1 else qubit_gates[0]
        else:
            # Apply default X gate to each qubit
            for qubit in self.auxiliary_qubits:
                gate = self.gate_types['x']()
                self.gates[qubit] = gate
                gate.apply(circuit, qubit)

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

    def _simulate(self):
        simulator = AerSimulator()
        result = simulator.run(self.circuit).result()
        counts = result.get_counts()
        return counts
    
    def run_simulation(self):
        # Get simulation results
        counts = self._simulate()
        
        # Results-based metrics
        results_metrics = {
            "counts": counts,
            "success_rate": counts.get('0' * self.payload_size, 0) / sum(counts.values()),
        }
        
        # Circuit metrics from Qiskit
        circuit_metrics = {
            "depth": self.circuit.depth(),
            "width": self.circuit.width(),
            "size": self.circuit.size(),
            "count_ops": self.circuit.count_ops(),
        }
        
        # Configuration metrics
        config_metrics = {
            "payload_size": self.payload_size,
        }
        
        # Custom gate distribution from our tracking
        gate_distribution = {}
        for qubit_gates in self.gates.values():
            if isinstance(qubit_gates, list):
                for gate in qubit_gates:
                    gate_distribution[gate.name] = gate_distribution.get(gate.name, 0) + 1
            else:
                # Single gate case
                gate_distribution[qubit_gates.name] = gate_distribution.get(qubit_gates.name, 0) + 1
        
        display(plot_histogram(counts))
        return {
            "results_metrics": results_metrics,
            "circuit_metrics": circuit_metrics,
            "config_metrics": config_metrics,
            "custom_gate_distribution": gate_distribution
        }

    def run_on_ibm(self, channel, token):
        try:
            QiskitRuntimeService.save_account(
                channel=channel, 
                token=token,
                overwrite=True
            )
            service = QiskitRuntimeService()
            backend = service.least_busy(simulator=False, operational=True)
            print(backend.configuration().backend_name)
            print(backend.status().pending_jobs)
            pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
            sampler = Sampler(backend)
            isa_circuit = pm.run(self.circuit)
            job = sampler.run([isa_circuit])
            print(f">>> Job ID: {job.job_id()}")
            print(f">>> Job Status: {job.status()}")

            import time
            start_time = time.time()
            while time.time() - start_time < 120:  # 2 minutes timeout
                status = job.status()
                print(f">>> Job Status: {status}")
                
                if status == 'DONE':
                    result = job.result()
                    counts = result[0].data.test_result.get_counts()
                    print(counts)
                    plot_histogram(counts)
                    return counts, self.circuit
                elif status != 'RUNNING' and status != 'QUEUED':
                    print(f"Job ended with status: {status}")
                    return 0, self.circuit
                
                time.sleep(5)

            print("Job timed out after 2 minutes")
            return 0, self.circuit
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0, self.circuit
        
    def get_ibm_job_results(self, job_id, token):
        # Get job results from the service
        service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance='ibm-q/open/main',
            token=token
        )
        job = service.job(job_id)
        job_result = job.result()
        
        # Get counts and plot histogram
        counts = job_result[0].data.test_result.get_counts()
        print("\nMeasurement Counts:")
        print(counts)
        display(plot_histogram(counts))
        
        # Get metrics from both the executed circuit and our original circuit
        print("\nCircuit Metrics:")
        print(f"Circuit depth: {self.circuit.depth()}")
        print(f"Circuit width: {self.circuit.width()}")
        print(f"Total number of operations: {self.circuit.size()}")
        print(f"Operation distribution: {self.circuit.count_ops()}")
        
        # Calculate success rate
        bit_string_length = len(next(iter(counts)))
        success_count = counts.get('0' * bit_string_length, 0)
        total_shots = sum(counts.values())
        success_rate = success_count / total_shots
        
        print(f"\nSuccess Rate: {success_rate:.2%}")
        
        # Return comprehensive metrics
        return {
            "counts": counts,
            "depth": self.circuit.depth(),
            "width": self.circuit.width(),
            "size": self.circuit.size(),
            "operations": self.circuit.count_ops(),
            "success_rate": success_rate,
            "job_result": job_result  # Include the full job result for additional analysis if needed
        }

class Experiments:
    def __init__(self):
        self.results = []
        self.validators = []  # Keep track of validators for later analysis
        
    def run_payload_size_gates_correlation(self, start: int = 1, end: int = 10):
        """
        Runs experiments increasing both payload_size and gates together.
        For example: (1,1), (2,2), ..., (10,10)
        """
        experiment_results = []
        
        for size in range(start, end + 1):
            print(f"\nRunning experiment with payload_size={size} and gates={size}")
            
            # Create validator
            validator = TeleportationValidator(
                payload_size=size,
                gates=size,  # This will create random gates
                use_barriers=True
            )
            
            # Draw circuit
            display(validator.draw())
            
            # Run simulation and get results
            results = validator.run_simulation()
            
            # Add experiment parameters to results
            results["experiment_params"] = {
                "experiment_type": "payload_size_gates_correlation",
                "payload_size": size,
                "num_gates": size
            }
            
            experiment_results.append(results)
            self.validators.append(validator)
            print(f"Experiment {size}/{end} completed")
            
        self.results.append({
            "name": "payload_size_gates_correlation",
            "experiments": experiment_results
        })
        
        return experiment_results
    
    def run_fixed_payload_varying_gates(self, payload_size: int, start_gates: int = 1, end_gates: int = 10):
        """
        Runs experiments with fixed payload_size but increasing number of gates
        """
        experiment_results = []
        
        for num_gates in range(start_gates, end_gates + 1):
            print(f"\nRunning experiment with payload_size={payload_size} and gates={num_gates}")
            
            # Create validator
            validator = TeleportationValidator(
                payload_size=payload_size,
                gates=num_gates,
                use_barriers=True
            )
            
            # Draw circuit
            display(validator.draw())
            
            # Run simulation
            results = validator.run_simulation()
            results["experiment_params"] = {
                "experiment_type": "fixed_payload_varying_gates",
                "payload_size": payload_size,
                "num_gates": num_gates
            }
            
            experiment_results.append(results)
            self.validators.append(validator)
            print(f"Experiment {num_gates}/{end_gates} completed")
            
        self.results.append({
            "name": "fixed_payload_varying_gates",
            "experiments": experiment_results
        })
        
        return experiment_results
    
    def plot_success_rates(self, experiment_name: str = None):
        """
        Plots success rates for experiments
        """
        import matplotlib.pyplot as plt
        
        # Collect rates directly from results
        rates = []
        for experiment_set in self.results:
            if experiment_name and experiment_set["name"] != experiment_name:
                continue
                
            for experiment in experiment_set["experiments"]:
                rates.append({
                    "type": experiment["experiment_params"]["experiment_type"],
                    "payload_size": experiment["experiment_params"]["payload_size"],
                    "num_gates": experiment["experiment_params"]["num_gates"],
                    "success_rate": experiment["results_metrics"]["success_rate"]
                })
        
        if not rates:
            print("No results to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Group by experiment type
        experiment_types = set(rate["type"] for rate in rates)
        
        for exp_type in experiment_types:
            type_rates = [r for r in rates if r["type"] == exp_type]
            
            if exp_type == "payload_size_gates_correlation":
                x_values = [r["payload_size"] for r in type_rates]
                label = "Payload Size (= Num Gates)"
            else:
                x_values = [r["num_gates"] for r in type_rates]
                label = "Number of Gates"
                
            y_values = [r["success_rate"] for r in type_rates]
            
            plt.plot(x_values, y_values, 'o-', label=exp_type)
            
        plt.xlabel(label)
        plt.ylabel("Success Rate")
        plt.title("Experiment Success Rates")
        plt.legend()
        plt.grid(True)
        plt.show()
