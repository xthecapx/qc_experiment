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
    def __init__(self, payload_size: int = 3, gates: list | int = None, use_barriers: bool = True, save_statevector: bool = False):
        self.gates = {}
        self.payload_size = payload_size
        self.use_barriers = use_barriers
        self.save_statevector = save_statevector
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
        if self.save_statevector:
            circuit.save_statevector(label='after_payload')
        if self.use_barriers:
            circuit.barrier()
    
        circuit = circuit.compose(
            self.protocol.circuit, 
            qubits=range(self.payload_size, self.payload_size + 3)
        )
        if self.use_barriers:
            circuit.barrier()
        
        if self.save_statevector:
            circuit.save_statevector(label='before_validation')
    
        self._create_validation(circuit)
        if self.save_statevector:
            circuit.save_statevector(label='after_validation')
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
        if self.save_statevector:
            simulator = AerSimulator(method='statevector')
        result = simulator.run(self.circuit).result()
        counts = result.get_counts()
        data = {'counts': counts}
        
        if self.save_statevector:
            data['after_payload'] = result.data()['after_payload']
            data['before_validation'] = result.data()['before_validation']
            data['after_validation'] = result.data()['after_validation']
            
        return data
    
    def run_simulation(self):
        # Get simulation results
        data = self._simulate()
        
        # Results-based metrics
        results_metrics = {
            "counts": data['counts'],
            "success_rate": data['counts'].get('0' * self.payload_size, 0) / sum(data['counts'].values()),
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
        
        display(plot_histogram(data['counts']))
        
        if self.save_statevector:
            print("\nState vector after payload:")
            display(data['after_payload'].draw('latex'))
            print("\nState vector before validation:")
            display(data['before_validation'].draw('latex'))
            print("\nState vector after validation:")
            display(data['after_validation'].draw('latex'))
        
        result = {
            "results_metrics": results_metrics,
            "circuit_metrics": circuit_metrics,
            "config_metrics": config_metrics,
            "custom_gate_distribution": gate_distribution
        }
        
        if self.save_statevector:
            result["statevector_data"] = {
                "after_payload": data['after_payload'],
                "before_validation": data['before_validation'],
                "after_validation": data['after_validation']
            }
            
        return result

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
            job_id = job.job_id()
            print(f">>> Job ID: {job_id}")
            print(f">>> Job Status: {job.status()}")

            import time
            start_time = time.time()
            while time.time() - start_time < 600:  # 10 minutes timeout
                status = job.status()
                print(f">>> Job Status: {status}")
                
                if status == 'DONE':
                    result = job.result()
                    counts = result[0].data.test_result.get_counts()
                    print(counts)
                    plot_histogram(counts)
                    return {
                        "status": "completed",
                        "job_id": job_id,
                        "counts": counts,
                        "backend": backend.configuration().backend_name
                    }
                elif status != 'RUNNING' and status != 'QUEUED':
                    print(f"Job ended with status: {status}")
                    return {
                        "status": "error",
                        "job_id": job_id,
                        "error": f"Job ended with status: {status}",
                        "backend": backend.configuration().backend_name
                    }
                
                time.sleep(5)

            print("Job timed out after 10 minutes")
            return {
                "status": "pending",
                "job_id": job_id,
                "backend": backend.configuration().backend_name
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
        
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
        self.validators = []
        
    def run_payload_size_gates_correlation(self, start: int = 1, end: int = 10, 
                                         run_on_ibm: bool = False, channel: str = None, token: str = None):
        experiment_results = []
        
        for size in range(start, end + 1):
            print(f"\nRunning experiment with payload_size={size} and gates={size}")
            
            use_barriers = not run_on_ibm
            validator = TeleportationValidator(
                payload_size=size,
                gates=size,
                use_barriers=use_barriers
            )
            
            display(validator.draw())
            
            # Determine actual execution type based on result
            execution_type = "simulation"
            if run_on_ibm and channel and token:
                ibm_result = validator.run_on_ibm(channel, token)
                if ibm_result["status"] == "completed" or ibm_result["status"] == "pending":
                    execution_type = "ibm"
                
                results = {
                    "status": ibm_result["status"],
                    "circuit_metrics": {
                        "depth": validator.circuit.depth(),
                        "width": validator.circuit.width(),
                        "size": validator.circuit.size(),
                        "count_ops": validator.circuit.count_ops(),
                    },
                    "config_metrics": {
                        "payload_size": size,
                    },
                    "ibm_data": ibm_result
                }
                
                if ibm_result["status"] == "completed":
                    results["results_metrics"] = {
                        "counts": ibm_result["counts"],
                        "success_rate": ibm_result["counts"].get('0' * size, 0) / sum(ibm_result["counts"].values()),
                    }
                elif ibm_result["status"] == "error":
                    # Fallback to simulation if IBM execution failed
                    results = validator.run_simulation()
                    results["status"] = "completed"
                    results["ibm_error"] = ibm_result.get("error")
            else:
                results = validator.run_simulation()
                results["status"] = "completed"
            
            results["experiment_params"] = {
                "experiment_type": "payload_size_gates_correlation",
                "payload_size": size,
                "num_gates": size,
                "execution_type": execution_type
            }
            
            experiment_results.append(results)
            self.validators.append(validator)
            print(f"Experiment {size}/{end} completed with status: {results['status']}")
        
        self.results.append({
            "name": "payload_size_gates_correlation",
            "experiments": experiment_results,
            "execution_type": "ibm" if any(r["experiment_params"]["execution_type"] == "ibm" 
                                         for r in experiment_results) else "simulation"
        })
        
        return experiment_results
    
    def run_fixed_payload_varying_gates(self, payload_size: int, start_gates: int = 1, end_gates: int = 10,
                                      run_on_ibm: bool = False, channel: str = None, token: str = None):
        experiment_results = []
        
        for num_gates in range(start_gates, end_gates + 1):
            print(f"\nRunning experiment with payload_size={payload_size} and gates={num_gates}")
            
            use_barriers = not run_on_ibm
            validator = TeleportationValidator(
                payload_size=payload_size,
                gates=num_gates,
                use_barriers=use_barriers
            )
            
            display(validator.draw())
            
            # Determine actual execution type based on result
            execution_type = "simulation"
            if run_on_ibm and channel and token:
                ibm_result = validator.run_on_ibm(channel, token)
                if ibm_result["status"] == "completed" or ibm_result["status"] == "pending":
                    execution_type = "ibm"
                
                results = {
                    "status": ibm_result["status"],
                    "circuit_metrics": {
                        "depth": validator.circuit.depth(),
                        "width": validator.circuit.width(),
                        "size": validator.circuit.size(),
                        "count_ops": validator.circuit.count_ops(),
                    },
                    "config_metrics": {
                        "payload_size": payload_size,
                    },
                    "ibm_data": ibm_result
                }
                
                if ibm_result["status"] == "completed":
                    results["results_metrics"] = {
                        "counts": ibm_result["counts"],
                        "success_rate": ibm_result["counts"].get('0' * payload_size, 0) / sum(ibm_result["counts"].values()),
                    }
                elif ibm_result["status"] == "error":
                    # Fallback to simulation if IBM execution failed
                    results = validator.run_simulation()
                    results["status"] = "completed"
                    results["ibm_error"] = ibm_result.get("error")
            else:
                results = validator.run_simulation()
                results["status"] = "completed"
            
            results["experiment_params"] = {
                "experiment_type": "fixed_payload_varying_gates",
                "payload_size": payload_size,
                "num_gates": num_gates,
                "execution_type": execution_type
            }
            
            experiment_results.append(results)
            self.validators.append(validator)
            print(f"Experiment {num_gates}/{end_gates} completed with status: {results['status']}")
        
        self.results.append({
            "name": "fixed_payload_varying_gates",
            "experiments": experiment_results,
            "execution_type": "ibm" if any(r["experiment_params"]["execution_type"] == "ibm" 
                                         for r in experiment_results) else "simulation"
        })
        
        return experiment_results

    def run_dynamic_payload_gates(self, payload_range: tuple, gates_range: tuple,
                                run_on_ibm: bool = False, channel: str = None, token: str = None):
        """
        Runs experiments with custom ranges for both payload size and number of gates.
        payload_range: tuple of (min_payload, max_payload)
        gates_range: tuple of (min_gates, max_gates)
        """
        experiment_results = []
        
        start_payload, end_payload = payload_range
        start_gates, end_gates = gates_range
        
        for payload_size in range(start_payload, end_payload + 1):
            for num_gates in range(start_gates, end_gates + 1):
                print(f"\nRunning experiment with payload_size={payload_size} and gates={num_gates}")
                
                use_barriers = not run_on_ibm
                validator = TeleportationValidator(
                    payload_size=payload_size,
                    gates=num_gates,
                    use_barriers=use_barriers
                )
                
                display(validator.draw())
                
                # Determine actual execution type based on result
                execution_type = "simulation"
                if run_on_ibm and channel and token:
                    ibm_result = validator.run_on_ibm(channel, token)
                    if ibm_result["status"] == "completed" or ibm_result["status"] == "pending":
                        execution_type = "ibm"
                    
                    results = {
                        "status": ibm_result["status"],
                        "circuit_metrics": {
                            "depth": validator.circuit.depth(),
                            "width": validator.circuit.width(),
                            "size": validator.circuit.size(),
                            "count_ops": validator.circuit.count_ops(),
                        },
                        "config_metrics": {
                            "payload_size": payload_size,
                        },
                        "ibm_data": ibm_result
                    }
                    
                    if ibm_result["status"] == "completed":
                        results["results_metrics"] = {
                            "counts": ibm_result["counts"],
                            "success_rate": ibm_result["counts"].get('0' * payload_size, 0) / sum(ibm_result["counts"].values()),
                        }
                    elif ibm_result["status"] == "error":
                        # Fallback to simulation if IBM execution failed
                        results = validator.run_simulation()
                        results["status"] = "completed"
                        results["ibm_error"] = ibm_result.get("error")
                else:
                    results = validator.run_simulation()
                    results["status"] = "completed"
                
                results["experiment_params"] = {
                    "experiment_type": "dynamic_payload_gates",
                    "payload_size": payload_size,
                    "num_gates": num_gates,
                    "execution_type": execution_type
                }
                
                experiment_results.append(results)
                self.validators.append(validator)
                print(f"Experiment with payload={payload_size}, gates={num_gates} completed with status: {results['status']}")
        
        self.results.append({
            "name": "dynamic_payload_gates",
            "experiments": experiment_results,
            "execution_type": "ibm" if any(r["experiment_params"]["execution_type"] == "ibm" 
                                         for r in experiment_results) else "simulation"
        })
        
        return experiment_results

    def plot_success_rates(self, experiment_name: str = None):
        """
        Plots success rates for experiments. If experiment_name is provided,
        only plots that specific experiment type.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Collect data from results
        data = {}
        for experiment_set in self.results:
            if experiment_name and experiment_set["name"] != experiment_name:
                continue
            
            for exp in experiment_set["experiments"]:
                if exp['status'] != 'completed':
                    continue
                
                payload = exp['experiment_params']['payload_size']
                gates = exp['experiment_params']['num_gates']
                # Convert success rate to percentage
                success = exp['results_metrics']['success_rate'] * 100
                
                if payload not in data:
                    data[payload] = {'gates': [], 'success': []}
                data[payload]['gates'].append(gates)
                data[payload]['success'].append(success)
        
        if not data:
            print("No results to plot")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot lines for each payload size with different colors and markers
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, (payload, values) in enumerate(sorted(data.items())):
            # Sort the data points by number of gates
            points = sorted(zip(values['gates'], values['success']))
            gates, success = zip(*points)
            
            plt.plot(gates, success, 
                     marker='o',
                     color=colors[i % len(colors)],
                     label=f'Payload Size {payload}')
        
        plt.xlabel('Number of Gates')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rates by Payload Size and Number of Gates')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        plt.show()
