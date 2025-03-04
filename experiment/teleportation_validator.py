from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import numpy as np
import random
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from results import results_1_4_2000_2005

# Load environment variables
load_dotenv()

# Get IBM Quantum credentials from environment variables
IBM_QUANTUM_CHANNEL = os.getenv('IBM_QUANTUM_CHANNEL', 'ibm_quantum')
IBM_QUANTUM_TOKEN = os.getenv('IBM_QUANTUM_TOKEN')

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
            "num_qubits": self.circuit.num_qubits,
            "num_clbits": self.circuit.num_clbits,
            "num_ancillas": self.circuit.num_ancillas,
            "num_parameters": self.circuit.num_parameters,
            "has_calibrations": bool(self.circuit.calibrations),
            "has_layout": bool(self.circuit.layout),
            "duration": self.circuit.estimate_duration() if hasattr(self.circuit, 'estimate_duration') else None
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

    def run_on_ibm(self, channel=None, token=None):
        try:
            # Use environment variables if no credentials are provided
            channel = channel or IBM_QUANTUM_CHANNEL
            token = token or IBM_QUANTUM_TOKEN
            
            if not token:
                raise ValueError("No IBM Quantum token provided. Please set IBM_QUANTUM_TOKEN in .env file or provide it directly.")

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
        
    def get_ibm_job_results(self, job_id):
        # Get job results from the service

        if not IBM_QUANTUM_TOKEN:
            raise ValueError("No IBM Quantum token provided. Please set IBM_QUANTUM_TOKEN in .env file or provide it directly.")
        
        print(IBM_QUANTUM_CHANNEL, job_id)
        service = QiskitRuntimeService(
            channel=IBM_QUANTUM_CHANNEL,
            instance='ibm-q/open/main',
            token=IBM_QUANTUM_TOKEN
        )
        job = service.job(job_id)
        job_result = job.result()
        
        # Get counts and plot histogram
        counts = job_result[0].data.test_result.get_counts()
        print("\nMeasurement Counts:")
        print(counts)
        display(plot_histogram(counts))
        
        # Calculate success rate
        bit_string_length = len(next(iter(counts)))
        success_count = counts.get('0' * bit_string_length, 0)
        total_shots = sum(counts.values())
        success_rate = success_count / total_shots
        
        print(f"\nSuccess Rate: {success_rate:.2%}")
        
        # Return comprehensive metrics matching _prepare_result_data
        return {
            "counts": counts,
            "success_rate": success_rate,
            "circuit_depth": self.circuit.depth(),
            "circuit_width": self.circuit.width(),
            "circuit_size": self.circuit.size(),
            "circuit_count_ops": self.circuit.count_ops(),
            "num_qubits": self.circuit.num_qubits,
            "num_clbits": self.circuit.num_clbits,
            "num_ancillas": self.circuit.num_ancillas,
            "num_parameters": self.circuit.num_parameters,
            "has_calibrations": bool(self.circuit.calibrations),
            "has_layout": bool(self.circuit.layout),
            "duration": self.circuit.estimate_duration() if hasattr(self.circuit, 'estimate_duration') else None,
            "job_result": job_result  # Include the full job result for additional analysis if needed
        }

class Experiments:
    def __init__(self):
        self.results_df = pd.DataFrame()
    
    def _serialize_dict(self, data):
        """Convert dictionary to JSON string"""
        return json.dumps(data)
    
    def _deserialize_dict(self, json_str):
        """Convert JSON string back to dictionary"""
        return json.loads(json_str) if pd.notna(json_str) else {}
    
    def _prepare_result_data(self, validator, status, execution_type, experiment_type, 
                           payload_size, num_gates, counts=None, success_rate=None,
                           ibm_data=None):
        """Helper method to prepare result data with proper serialization"""
        result_data = {
            "status": status,
            "circuit_depth": validator.circuit.depth(),
            "circuit_width": validator.circuit.width(),
            "circuit_size": validator.circuit.size(),
            "circuit_count_ops": self._serialize_dict(validator.circuit.count_ops()),
            "num_qubits": validator.circuit.num_qubits,
            "num_clbits": validator.circuit.num_clbits,
            "num_ancillas": validator.circuit.num_ancillas,
            "num_parameters": validator.circuit.num_parameters,
            "has_calibrations": bool(validator.circuit.calibrations),
            "has_layout": bool(validator.circuit.layout),
            "duration": validator.circuit.estimate_duration() if hasattr(validator.circuit, 'estimate_duration') else None,
            "payload_size": payload_size,
            "num_gates": num_gates,
            "execution_type": execution_type,
            "experiment_type": experiment_type
        }
        
        if counts is not None:
            result_data["counts"] = self._serialize_dict(counts)
        if success_rate is not None:
            result_data["success_rate"] = success_rate
        if ibm_data:
            result_data.update({
                "ibm_job_id": ibm_data.get("job_id"),
                "ibm_backend": ibm_data.get("backend")
            })
        
        return result_data

    def run_payload_size_gates_correlation(self, start: int = 1, end: int = 10, 
                                         run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                         show_circuit: bool = False):
        for size in range(start, end + 1):
            print(f"\nRunning experiment with payload_size={size} and gates={size}")
            
            use_barriers = not run_on_ibm
            validator = TeleportationValidator(
                payload_size=size,
                gates=size,
                use_barriers=use_barriers
            )
            
            if show_circuit:
                display(validator.draw())
            
            # Determine actual execution type based on result
            execution_type = "simulation"
            if run_on_ibm and channel and token:
                ibm_result = validator.run_on_ibm(channel, token)
                if ibm_result["status"] == "completed" or ibm_result["status"] == "pending":
                    execution_type = "ibm"
                
                result_data = self._prepare_result_data(
                    validator=validator,
                    status=ibm_result["status"],
                    execution_type=execution_type,
                    experiment_type="payload_size_gates_correlation",
                    payload_size=size,
                    num_gates=size
                )
                
                if ibm_result["status"] == "completed":
                    result_data.update({
                        "counts": self._serialize_dict(ibm_result["counts"]),
                        "success_rate": ibm_result["counts"].get('0' * size, 0) / sum(ibm_result["counts"].values()),
                        "ibm_job_id": ibm_result.get("job_id"),
                        "ibm_backend": ibm_result.get("backend")
                    })
                elif ibm_result["status"] == "error":
                    # Fallback to simulation if IBM execution failed
                    sim_result = validator.run_simulation()
                    result_data.update({
                        "status": "completed",
                        "counts": self._serialize_dict(sim_result["results_metrics"]["counts"]),
                        "success_rate": sim_result["results_metrics"]["success_rate"],
                        "ibm_error": ibm_result.get("error")
                    })
            else:
                sim_result = validator.run_simulation()
                result_data = self._prepare_result_data(
                    validator=validator,
                    status="completed",
                    execution_type="simulation",
                    experiment_type="payload_size_gates_correlation",
                    payload_size=size,
                    num_gates=size,
                    counts=sim_result["results_metrics"]["counts"],
                    success_rate=sim_result["results_metrics"]["success_rate"]
                )
            
            # Append to DataFrame
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
            print(f"Experiment {size}/{end} completed with status: {result_data['status']}")
        
        # Export results to CSV with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self.export_to_csv(f'experiment_results_payload_correlation_{timestamp}.csv')
        
        return self.results_df

    def run_fixed_payload_varying_gates(self, payload_size: int, start_gates: int = 1, end_gates: int = 10,
                                      run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                      show_circuit: bool = False):
        for num_gates in range(start_gates, end_gates + 1):
            print(f"\nRunning experiment with payload_size={payload_size} and gates={num_gates}")
            
            use_barriers = not run_on_ibm
            validator = TeleportationValidator(
                payload_size=payload_size,
                gates=num_gates,
                use_barriers=use_barriers
            )
            
            if show_circuit:
                display(validator.draw())
            
            # Determine actual execution type based on result
            execution_type = "simulation"
            if run_on_ibm and channel and token:
                ibm_result = validator.run_on_ibm(channel, token)
                if ibm_result["status"] == "completed" or ibm_result["status"] == "pending":
                    execution_type = "ibm"
                
                result_data = {
                    "status": ibm_result["status"],
                    "circuit_depth": validator.circuit.depth(),
                    "circuit_width": validator.circuit.width(),
                    "circuit_size": validator.circuit.size(),
                    "circuit_count_ops": str(validator.circuit.count_ops()),
                    "payload_size": payload_size,
                    "num_gates": num_gates,
                    "execution_type": execution_type,
                    "experiment_type": "fixed_payload_varying_gates"
                }
                
                if ibm_result["status"] == "completed":
                    result_data.update({
                        "counts": str(ibm_result["counts"]),
                        "success_rate": ibm_result["counts"].get('0' * payload_size, 0) / sum(ibm_result["counts"].values()),
                        "ibm_job_id": ibm_result.get("job_id"),
                        "ibm_backend": ibm_result.get("backend")
                    })
                elif ibm_result["status"] == "error":
                    # Fallback to simulation if IBM execution failed
                    sim_result = validator.run_simulation()
                    result_data.update({
                        "status": "completed",
                        "counts": str(sim_result["results_metrics"]["counts"]),
                        "success_rate": sim_result["results_metrics"]["success_rate"],
                        "ibm_error": ibm_result.get("error")
                    })
            else:
                sim_result = validator.run_simulation()
                result_data = {
                    "status": "completed",
                    "circuit_depth": validator.circuit.depth(),
                    "circuit_width": validator.circuit.width(),
                    "circuit_size": validator.circuit.size(),
                    "circuit_count_ops": str(validator.circuit.count_ops()),
                    "payload_size": payload_size,
                    "num_gates": num_gates,
                    "execution_type": "simulation",
                    "experiment_type": "fixed_payload_varying_gates",
                    "counts": str(sim_result["results_metrics"]["counts"]),
                    "success_rate": sim_result["results_metrics"]["success_rate"]
                }
            
            # Append to DataFrame
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
            print(f"Experiment {num_gates}/{end_gates} completed with status: {result_data['status']}")
        
        # Export results to CSV with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self.export_to_csv(f'experiment_results_fixed_payload_{payload_size}_{timestamp}.csv')
        
        return self.results_df

    def run_dynamic_payload_gates(self, payload_range: tuple, gates_range: tuple,
                                run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                show_circuit: bool = False):
        """
        Runs experiments with custom ranges for both payload size and number of gates.
        payload_range: tuple of (min_payload, max_payload)
        gates_range: tuple of (min_gates, max_gates)
        """
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
                
                if show_circuit:
                    display(validator.draw())
                
                # Determine actual execution type based on result
                execution_type = "simulation"
                if run_on_ibm and channel and token:
                    ibm_result = validator.run_on_ibm(channel, token)
                    if ibm_result["status"] == "completed" or ibm_result["status"] == "pending":
                        execution_type = "ibm"
                    
                    result_data = self._prepare_result_data(
                        validator=validator,
                        status=ibm_result["status"],
                        execution_type=execution_type,
                        experiment_type="dynamic_payload_gates",
                        payload_size=payload_size,
                        num_gates=num_gates
                    )
                    
                    if ibm_result["status"] == "completed":
                        result_data.update({
                            "counts": self._serialize_dict(ibm_result["counts"]),
                            "success_rate": ibm_result["counts"].get('0' * payload_size, 0) / sum(ibm_result["counts"].values()),
                            "ibm_job_id": ibm_result.get("job_id"),
                            "ibm_backend": ibm_result.get("backend")
                        })
                    elif ibm_result["status"] == "error":
                        # Fallback to simulation if IBM execution failed
                        sim_result = validator.run_simulation()
                        result_data.update({
                            "status": "completed",
                            "counts": self._serialize_dict(sim_result["results_metrics"]["counts"]),
                            "success_rate": sim_result["results_metrics"]["success_rate"],
                            "ibm_error": ibm_result.get("error")
                        })
                else:
                    sim_result = validator.run_simulation()
                    result_data = self._prepare_result_data(
                        validator=validator,
                        status="completed",
                        execution_type="simulation",
                        experiment_type="dynamic_payload_gates",
                        payload_size=payload_size,
                        num_gates=num_gates,
                        counts=sim_result["results_metrics"]["counts"],
                        success_rate=sim_result["results_metrics"]["success_rate"]
                    )
                
                # Append to DataFrame
                self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
                print(f"Experiment with payload={payload_size}, gates={num_gates} completed with status: {result_data['status']}")
        
        # Export results to CSV with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self.export_to_csv(f'experiment_results_dynamic_{start_payload}-{end_payload}_{start_gates}-{end_gates}_{timestamp}.csv')
        
        return self.results_df

    def plot_success_rates(self, experiment_name: str = None):
        """
        Plots success rates for experiments. If experiment_name is provided,
        only plots that specific experiment type.
        """
        # Filter DataFrame if experiment_name is provided
        df = self.results_df
        if experiment_name:
            df = df[df['experiment_type'] == experiment_name]
        
        if df.empty:
            print("No results to plot")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot lines for each payload size
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, payload in enumerate(sorted(df['payload_size'].unique())):
            payload_data = df[df['payload_size'] == payload]
            plt.plot(payload_data['num_gates'], 
                    payload_data['success_rate'] * 100,
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

    def export_to_csv(self, filename: str = 'experiment_results.csv'):
        """
        Exports the experiment results to a CSV file.
        Args:
            filename (str): Name of the CSV file to save the results
        """
        self.results_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    @classmethod
    def from_csv(cls, filename: str = 'experiment_results.csv'):
        """
        Creates an Experiments instance from a CSV file.
        Args:
            filename (str): Name of the CSV file to read the results from
        Returns:
            Experiments: New instance with loaded results
        """
        # Create new instance
        experiments = cls()
        
        # Read the CSV file directly into the DataFrame
        experiments.results_df = pd.read_csv(filename)
        
        # Convert JSON strings back to dictionaries for specific columns
        json_columns = ['circuit_count_ops', 'counts']
        for col in json_columns:
            if col in experiments.results_df.columns:
                experiments.results_df[col] = experiments.results_df[col].apply(experiments._deserialize_dict)
        
        return experiments

    def get_circuit_operations(self, row_index: int = None):
        """
        Get circuit operations as a dictionary.
        Args:
            row_index (int, optional): Index of the row to get operations from. If None, returns all.
        Returns:
            dict or list of dicts: Circuit operations
        """
        if row_index is not None:
            return self._deserialize_dict(self.results_df.iloc[row_index]['circuit_count_ops'])
        return [self._deserialize_dict(ops) for ops in self.results_df['circuit_count_ops']]

    def get_counts(self, row_index: int = None):
        """
        Get measurement counts as a dictionary.
        Args:
            row_index (int, optional): Index of the row to get counts from. If None, returns all.
        Returns:
            dict or list of dicts: Measurement counts
        """
        if row_index is not None:
            return self._deserialize_dict(self.results_df.iloc[row_index]['counts'])
        return [self._deserialize_dict(counts) for counts in self.results_df['counts']]
