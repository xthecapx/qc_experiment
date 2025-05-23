from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
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
# from results import results_1_4_2000_2005
from qiskit.providers.fake_provider import GenericBackendV2

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
        if self.name == 's':
            qc.sdg(qubit)  # S-dagger is the conjugate of S
        elif self.name == 'sdg':
            qc.s(qubit)    # S is the conjugate of S-dagger
        else:
            # x, z, h, y are self-inverse
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
            'x': lambda: QuantumGate('x'),  # Pauli-X (self-inverse)
            'y': lambda: QuantumGate('y'),  # Pauli-Y (self-inverse)
            'z': lambda: QuantumGate('z'),  # Pauli-Z (self-inverse)
            'h': lambda: QuantumGate('h'),  # Hadamard (self-inverse)
            's': lambda: QuantumGate('s'),  # Phase gate
            'sdg': lambda: QuantumGate('sdg'),  # S-dagger gate (inverse of S)
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
    
    def run_simulation(self, show_histogram=True):
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
            # "duration": self.circuit.estimate_duration() if hasattr(self.circuit, 'estimate_duration') else None
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
        
        if show_histogram:
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
            "duration": job.usage_estimation,
            "job_result": job_result
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
                "ibm_backend": ibm_data.get("backend"),
                "job_duration": ibm_data.get("metrics", {}).get("usage", {}).get("seconds", 0.0),
                "job_quantum_duration": ibm_data.get("metrics", {}).get("usage", {}).get("quantum_seconds", 0.0),
                "job_status": ibm_data.get("status", "error"),
                "job_created": ibm_data.get("metrics", {}).get("timestamps", {}).get("created"),
                "job_finished": ibm_data.get("metrics", {}).get("timestamps", {}).get("finished"),
                "job_running": ibm_data.get("metrics", {}).get("timestamps", {}).get("running")
            })
        
        return result_data

    def run_payload_size_gates_correlation(self, start: int = 1, end: int = 10, 
                                         run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                         show_circuit: bool = False):
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_payload_correlation_{timestamp}.csv'
        
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
            
            # Save after each iteration to ensure data is not lost
            self.export_to_csv(output_file)
        
        return self.results_df

    def run_fixed_payload_varying_gates(self, payload_size: int, start_gates: int = 1, end_gates: int = 10,
                                      run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                      show_circuit: bool = False):
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_fixed_payload_{payload_size}_{timestamp}.csv'
        
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
            
            # Save after each iteration to ensure data is not lost
            self.export_to_csv(output_file)
        
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
        
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_dynamic_{start_payload}-{end_payload}_{start_gates}-{end_gates}_{timestamp}.csv'
        
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
                
                # Save after each iteration to ensure data is not lost
                self.export_to_csv(output_file)
        
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
        This version overwrites the file after each iteration to ensure data is not lost.
        
        Args:
            filename (str): Name of the CSV file to save the results
        """
        # If file exists, check for header
        header = True
        if os.path.exists(filename):
            header = False
        
        # Write the DataFrame to CSV (mode='w' to overwrite the file)
        self.results_df.to_csv(filename, index=False, mode='w')
        print(f"Results exported to {filename}")

    def run_controlled_depth_experiment(self, payload_sizes: list = [1, 2, 3, 4, 5], max_depth: int = 5, 
                                      run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                      show_circuit: bool = False, show_histogram: bool = False):
        """
        Run an experiment with controlled circuit depth.
        
        For each payload size, the number of gates increases proportionally to maintain a controlled depth increase:
        - For payload_size=1: gates increase by 1 per experiment
        - For payload_size=2: gates increase by 2 per experiment
        - For payload_size=3: gates increase by 3 per experiment
        And so on.
        
        The base circuit depth is approximately 9 (with payload_size=1 and gates=1).
        
        Args:
            payload_sizes: List of payload sizes to test
            max_depth: Number of depth increments to test for each payload size
            run_on_ibm: Whether to run on IBM quantum hardware
            channel: IBM Quantum channel
            token: IBM Quantum token
            show_circuit: Whether to display the circuit diagram
            show_histogram: Whether to display histograms of measurement results
        """
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_controlled_depth_{timestamp}.csv'
        
        for payload_size in payload_sizes:
            print(f"\nRunning experiments for payload_size={payload_size}")
            
            for depth_increment in range(0, max_depth + 1):
                # Calculate gates based on payload size and depth increment
                gates = depth_increment * payload_size
                
                print(f"  Running experiment with payload_size={payload_size}, gates={gates}")
                
                use_barriers = not run_on_ibm
                validator = TeleportationValidator(
                    payload_size=payload_size,
                    gates=gates,
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
                        experiment_type="controlled_depth_experiment",
                        payload_size=payload_size,
                        num_gates=gates
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
                        sim_result = validator.run_simulation(show_histogram=show_histogram)
                        result_data.update({
                            "status": "completed",
                            "counts": self._serialize_dict(sim_result["results_metrics"]["counts"]),
                            "success_rate": sim_result["results_metrics"]["success_rate"],
                            "ibm_error": ibm_result.get("error")
                        })
                else:
                    sim_result = validator.run_simulation(show_histogram=show_histogram)
                    result_data = self._prepare_result_data(
                        validator=validator,
                        status="completed",
                        execution_type="simulation",
                        experiment_type="controlled_depth_experiment",
                        payload_size=payload_size,
                        num_gates=gates,
                        counts=sim_result["results_metrics"]["counts"],
                        success_rate=sim_result["results_metrics"]["success_rate"]
                    )
                
                # Append to DataFrame
                self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
                print(f"  Experiment completed with status: {result_data['status']}, circuit depth: {validator.circuit.depth()}")
                
                # Save after each iteration to ensure data is not lost
                self.export_to_csv(output_file)
        
        return self.results_df

    def run_target_depth_experiment(self, target_depths: list = None, max_payload_size: int = 5,
                                  run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                  show_circuit: bool = False, show_histogram: bool = False, min_experiments_per_depth: int = 5):
        """
        Run experiments with specific target circuit depths using various combinations of payload size and gates.
        
        For each target depth, this method generates combinations of payload size and gates that should
        produce that depth, then runs experiments for each combination to compare success rates.
        
        If there are fewer valid combinations than min_experiments_per_depth, it will duplicate the valid
        combinations to reach the desired number of experiments.
        
        Based on empirical results, the actual depth formula is:
        - Base depth with 0 gates = 13 + 2 * (payload_size - 1)
        - Depth with gates = base_depth + 2 * (gates / payload_size) - 2
        
        Args:
            target_depths: List of specific circuit depths to target (if None, uses [13, 15, 17, ..., 49])
            max_payload_size: Maximum payload size to consider
            run_on_ibm: Whether to run on IBM quantum hardware
            channel: IBM Quantum channel
            token: IBM Quantum token
            show_circuit: Whether to display the circuit diagram
            show_histogram: Whether to display histograms of measurement results
            min_experiments_per_depth: Minimum number of different combinations to run for each depth
        """
        if target_depths is None:
            target_depths = list(range(13, 50, 2))  # [13, 15, 17, ..., 49]
        
        # Generate combinations for each target depth
        all_combinations = []
        for depth in target_depths:
            valid_combinations = []
            
            # Find valid combinations that produce the exact target depth
            for payload_size in range(1, max_payload_size + 1):
                # Calculate base depth for this payload size
                base_depth = 13 + 2 * (payload_size - 1)
                
                # If target depth is less than base depth, skip this payload size
                if depth < base_depth:
                    continue
                
                # Calculate required gates to achieve target depth
                # Corrected formula based on empirical results:
                # depth = base_depth + 2 * (gates / payload_size) - 2
                # Solving for gates: gates = (depth - base_depth + 2) * payload_size / 2
                required_gates = (depth - base_depth + 2) * payload_size / 2
                
                # Only add if required_gates is a non-negative integer
                if required_gates >= 0 and required_gates.is_integer():
                    valid_combinations.append((payload_size, int(required_gates)))
            
            # If we have valid combinations, duplicate them to reach min_experiments_per_depth
            if valid_combinations:
                # Duplicate combinations if needed
                combinations_to_run = []
                while len(combinations_to_run) < min_experiments_per_depth:
                    for combo in valid_combinations:
                        if len(combinations_to_run) < min_experiments_per_depth:
                            combinations_to_run.append(combo)
                        else:
                            break
                
                all_combinations.append((depth, combinations_to_run))
        
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_target_depth_{timestamp}.csv'
        
        # Run experiments for each valid combination
        for depth, combinations in all_combinations:
            print(f"\nRunning experiments for target depth: {depth} ({len(combinations)} combinations)")
            
            for payload_size, gates in combinations:
                print(f"  Running experiment with payload_size={payload_size}, gates={gates}")
                
                use_barriers = not run_on_ibm
                validator = TeleportationValidator(
                    payload_size=payload_size,
                    gates=gates,
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
                        experiment_type="target_depth_experiment",
                        payload_size=payload_size,
                        num_gates=gates
                    )
                    
                    if ibm_result["status"] == "completed":
                        result_data.update({
                            "counts": self._serialize_dict(ibm_result["counts"]),
                            "success_rate": ibm_result["counts"].get('0' * payload_size, 0) / sum(ibm_result["counts"].values()),
                            "ibm_job_id": ibm_result.get("job_id"),
                            "ibm_backend": ibm_result.get("backend"),
                            "target_depth": depth
                        })
                    elif ibm_result["status"] == "error":
                        # Fallback to simulation if IBM execution failed
                        sim_result = validator.run_simulation(show_histogram=show_histogram)
                        result_data.update({
                            "status": "completed",
                            "counts": self._serialize_dict(sim_result["results_metrics"]["counts"]),
                            "success_rate": sim_result["results_metrics"]["success_rate"],
                            "ibm_error": ibm_result.get("error"),
                            "target_depth": depth
                        })
                else:
                    sim_result = validator.run_simulation(show_histogram=show_histogram)
                    result_data = self._prepare_result_data(
                        validator=validator,
                        status="completed",
                        execution_type="simulation",
                        experiment_type="target_depth_experiment",
                        payload_size=payload_size,
                        num_gates=gates,
                        counts=sim_result["results_metrics"]["counts"],
                        success_rate=sim_result["results_metrics"]["success_rate"]
                    )
                    result_data["target_depth"] = depth
                
                # Append to DataFrame
                self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
                actual_depth = validator.circuit.depth()
                print(f"  Experiment completed with status: {result_data['status']}, target depth: {depth}, actual depth: {actual_depth}")
                
                # Save after each iteration to ensure data is not lost
                self.export_to_csv(output_file)
                
                # Verify if actual depth matches target depth
                if actual_depth != depth:
                    print(f"  WARNING: Actual depth {actual_depth} does not match target depth {depth}")
        
        return self.results_df

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

    def update_table_with_job_info(self, input_csv: str, output_csv: str = None):
        """
        Updates the experiment results table with IBM job information.
        Args:
            input_csv (str): Path to the input CSV file
            output_csv (str, optional): Path to save the updated CSV. If None, will append '_updated' to input filename
        Returns:
            pd.DataFrame: Updated DataFrame with job information
        """
        # Read the input CSV
        df = pd.read_csv(input_csv)
        
        # Remove empty duration column if it exists
        if 'duration' in df.columns:
            df = df.drop('duration', axis=1)
        
        # Create output filename if not provided
        if output_csv is None:
            base_name = os.path.splitext(input_csv)[0]
            output_csv = f"{base_name}_updated.csv"
        
        # Create a single service instance
        if not IBM_QUANTUM_TOKEN:
            raise ValueError("No IBM Quantum token provided. Please set IBM_QUANTUM_TOKEN in .env file or provide it directly.")
        
        service = QiskitRuntimeService(
            channel=IBM_QUANTUM_CHANNEL,
            instance='ibm-q/open/main',
            token=IBM_QUANTUM_TOKEN
        )
        
        # Initialize new columns with appropriate data types
        df['job_duration'] = pd.Series(dtype='float64')
        df['job_quantum_duration'] = pd.Series(dtype='float64')
        df['job_status'] = pd.Series(dtype='object')
        df['job_success_rate'] = pd.Series(dtype='float64')
        df['job_created'] = pd.Series(dtype='object')
        df['job_finished'] = pd.Series(dtype='object')
        df['job_running'] = pd.Series(dtype='object')
        df['job_execution_spans'] = pd.Series(dtype='object')
        df['job_execution_duration'] = pd.Series(dtype='float64')
        
        # Process each row with an IBM job ID
        for idx, row in df.iterrows():
            if pd.notna(row.get('ibm_job_id')):
                try:
                    # Get job using the service instance
                    job = service.job(row['ibm_job_id'])
                    job_result = job.result()
                    
                    # Get metrics
                    metrics = job.metrics()
                    
                    # Get counts and calculate success rate
                    counts = job_result[0].data.test_result.get_counts()
                    bit_string_length = len(next(iter(counts)))
                    success_count = counts.get('0' * bit_string_length, 0)
                    total_shots = sum(counts.values())
                    success_rate = success_count / total_shots
                    
                    # Get execution spans from result metadata
                    execution_spans = job_result.metadata.get('execution', {}).get('execution_spans', [])
                    
                    # Calculate execution duration from spans
                    execution_duration = 0.0
                    if execution_spans:
                        for span in execution_spans:
                            # Access start and stop directly from the span object
                            start_time = pd.Timestamp(span.start)
                            stop_time = pd.Timestamp(span.stop)
                            execution_duration += (stop_time - start_time).total_seconds()
                    
                    # Update the row with job information
                    df.at[idx, 'job_duration'] = metrics.get('usage', {}).get('seconds', 0.0)
                    df.at[idx, 'job_quantum_duration'] = metrics.get('usage', {}).get('quantum_seconds', 0.0)
                    df.at[idx, 'job_status'] = 'completed'
                    df.at[idx, 'job_success_rate'] = success_rate
                    df.at[idx, 'job_created'] = metrics.get('timestamps', {}).get('created')
                    df.at[idx, 'job_finished'] = metrics.get('timestamps', {}).get('finished')
                    df.at[idx, 'job_running'] = metrics.get('timestamps', {}).get('running')
                    df.at[idx, 'job_execution_spans'] = str(execution_spans)
                    df.at[idx, 'job_execution_duration'] = execution_duration
                    
                    # Update missing information in the original columns
                    if pd.isna(row.get('status')) or row.get('status') == 'pending':
                        df.at[idx, 'status'] = 'completed'
                    
                    if pd.isna(row.get('counts')) or row.get('counts') == '':
                        df.at[idx, 'counts'] = self._serialize_dict(counts)
                    
                    if pd.isna(row.get('success_rate')):
                        df.at[idx, 'success_rate'] = success_rate
                    
                    if pd.isna(row.get('ibm_backend')) or row.get('ibm_backend') == '':
                        df.at[idx, 'ibm_backend'] = job.backend().name
                    
                    # Update target_depth if missing
                    if 'target_depth' in df.columns and (pd.isna(row.get('target_depth')) or row.get('target_depth') == ''):
                        # Try to infer target_depth from other rows with similar payload and gates
                        similar_rows = df[(df['payload_size'] == row['payload_size']) & 
                                         (df['num_gates'] == row['num_gates']) & 
                                         pd.notna(df['target_depth'])]
                        
                        if not similar_rows.empty:
                            # Use the most common target depth value for similar configurations
                            df.at[idx, 'target_depth'] = similar_rows['target_depth'].mode()[0]
                        else:
                            # If no similar rows found, calculate depth based on empirical formula
                            # Formula from the comments in run_target_depth_experiment:
                            # depth = base_depth + 2 * (gates / payload_size) - 2
                            # base_depth = 13 + 2 * (payload_size - 1)
                            base_depth = 13 + 2 * (row['payload_size'] - 1)
                            calculated_depth = base_depth + 2 * (row['num_gates'] / row['payload_size']) - 2
                            df.at[idx, 'target_depth'] = calculated_depth
                    
                    print(f"Updated job {row['ibm_job_id']} information")
                except Exception as e:
                    print(f"Error processing job {row['ibm_job_id']}: {str(e)}")
                    df.at[idx, 'job_status'] = 'error'
                    df.at[idx, 'job_duration'] = 0.0
                    df.at[idx, 'job_quantum_duration'] = 0.0
                    df.at[idx, 'job_success_rate'] = 0.0
                    df.at[idx, 'job_execution_duration'] = 0.0
                    
                    # If status is still pending, mark as error
                    if row.get('status') == 'pending':
                        df.at[idx, 'status'] = 'error'
        
        # Save the updated DataFrame
        df.to_csv(output_csv, index=False)
        print(f"Updated results saved to {output_csv}")
        
        return df

    def create_table_from_results(self, results_list: list):
        """
        Creates a DataFrame from experiment results and adds IBM job information.
        Args:
            results_list (list): List of experiment results from the Python file
        Returns:
            pd.DataFrame: DataFrame with job information
        """
        # Create DataFrame from results
        rows = []
        for group in results_list:
            for experiment in group['experiments']:
                row = {
                    'status': experiment['status'],
                    'circuit_depth': experiment['circuit_metrics']['depth'],
                    'circuit_width': experiment['circuit_metrics']['width'],
                    'circuit_size': experiment['circuit_metrics']['size'],
                    'circuit_count_ops': json.dumps(experiment['circuit_metrics']['count_ops']),
                    'payload_size': experiment['config_metrics']['payload_size'],
                    'num_gates': experiment['experiment_params']['num_gates'],
                    'execution_type': experiment['experiment_params']['execution_type'],
                    'experiment_type': experiment['experiment_params']['experiment_type'],
                    'ibm_job_id': experiment.get('ibm_data', {}).get('job_id'),
                    'ibm_backend': experiment.get('ibm_data', {}).get('backend'),
                    'counts': json.dumps(experiment.get('results_metrics', {}).get('counts', {})),
                    'success_rate': experiment.get('results_metrics', {}).get('success_rate', 0.0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create a single service instance
        if not IBM_QUANTUM_TOKEN:
            raise ValueError("No IBM Quantum token provided. Please set IBM_QUANTUM_TOKEN in .env file or provide it directly.")
        
        service = QiskitRuntimeService(
            channel=IBM_QUANTUM_CHANNEL,
            instance='ibm-q/open/main',
            token=IBM_QUANTUM_TOKEN
        )
        
        # Initialize new columns with appropriate data types
        df['job_duration'] = pd.Series(dtype='float64')
        df['job_quantum_duration'] = pd.Series(dtype='float64')
        df['job_status'] = pd.Series(dtype='object')
        df['job_success_rate'] = pd.Series(dtype='float64')
        df['job_created'] = pd.Series(dtype='object')
        df['job_finished'] = pd.Series(dtype='object')
        df['job_running'] = pd.Series(dtype='object')
        df['job_execution_spans'] = pd.Series(dtype='object')
        df['job_execution_duration'] = pd.Series(dtype='float64')
        
        # Process each row with an IBM job ID
        for idx, row in df.iterrows():
            if pd.notna(row.get('ibm_job_id')):
                try:
                    # Get job using the service instance
                    job = service.job(row['ibm_job_id'])
                    job_result = job.result()
                    
                    # Get metrics
                    metrics = job.metrics()
                    
                    # Get execution spans from result metadata
                    execution_spans = job_result.metadata.get('execution', {}).get('execution_spans', [])
                    
                    # Calculate execution duration from spans
                    execution_duration = 0.0
                    if execution_spans:
                        for span in execution_spans:
                            # Access start and stop directly from the span object
                            start_time = pd.Timestamp(span.start)
                            stop_time = pd.Timestamp(span.stop)
                            execution_duration += (stop_time - start_time).total_seconds()
                    
                    # Update the row with job information
                    df.at[idx, 'job_duration'] = metrics.get('usage', {}).get('seconds', 0.0)
                    df.at[idx, 'job_quantum_duration'] = metrics.get('usage', {}).get('quantum_seconds', 0.0)
                    df.at[idx, 'job_status'] = 'completed'
                    df.at[idx, 'job_success_rate'] = row['success_rate']
                    df.at[idx, 'job_created'] = metrics.get('timestamps', {}).get('created')
                    df.at[idx, 'job_finished'] = metrics.get('timestamps', {}).get('finished')
                    df.at[idx, 'job_running'] = metrics.get('timestamps', {}).get('running')
                    df.at[idx, 'job_execution_spans'] = str(execution_spans)
                    df.at[idx, 'job_execution_duration'] = execution_duration
                    
                    print(f"Updated job {row['ibm_job_id']} information")
                except Exception as e:
                    print(f"Error processing job {row['ibm_job_id']}: {str(e)}")
                    df.at[idx, 'job_status'] = 'error'
                    df.at[idx, 'job_duration'] = 0.0
                    df.at[idx, 'job_quantum_duration'] = 0.0
                    df.at[idx, 'job_success_rate'] = 0.0
                    df.at[idx, 'job_execution_duration'] = 0.0
        
        # Get payload and gates ranges from the data
        min_payload = df['payload_size'].min()
        max_payload = df['payload_size'].max()
        min_gates = df['num_gates'].min()
        max_gates = df['num_gates'].max()
        
        # Export to CSV with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_csv = f'experiment_results_dynamic_{min_payload}-{max_payload}_{min_gates}-{max_gates}_{timestamp}.csv'
        df.to_csv(output_csv, index=False)
        print(f"Results exported to {output_csv}")
        
        return df

    def run_fixed_payload_experiments(self, payload_gates_map: dict = None, iterations: int = 10,
                                  run_on_ibm: bool = False, channel: str = IBM_QUANTUM_CHANNEL, token: str = IBM_QUANTUM_TOKEN,
                                  show_circuit: bool = True, show_histogram: bool = False):
        """
        Run experiments with fixed payload sizes and corresponding number of gates.
        
        For each payload size, runs the specified number of iterations with the specified number of gates.
        Default configuration follows the pattern:
        - For payload_size=1: 10 experiments with 3 gates
        - For payload_size=2: 10 experiments with 6 gates 
        - For payload_size=3: 10 experiments with 9 gates
        - For payload_size=4: 10 experiments with 12 gates
        - For payload_size=5: 10 experiments with 15 gates
        
        Args:
            payload_gates_map: Dictionary mapping payload sizes to number of gates
                              (default: {1:3, 2:6, 3:9, 4:12, 5:15})
            iterations: Number of experiments to run for each payload size
            run_on_ibm: Whether to run on IBM quantum hardware
            channel: IBM Quantum channel
            token: IBM Quantum token
            show_circuit: Whether to display the circuit diagram
            show_histogram: Whether to display histograms of measurement results
        """
        # Default payload-gates map if not provided
        if payload_gates_map is None:
            payload_gates_map = {
                1: 3,
                2: 6,
                3: 9,
                4: 12,
                5: 15
            }
        
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_fixed_payload_{timestamp}.csv'
        
        # Run experiments for each payload size and gates configuration
        for payload_size, num_gates in payload_gates_map.items():
            print(f"\nRunning {iterations} experiments with payload_size={payload_size}, gates={num_gates}")
            
            for iteration in range(1, iterations + 1):
                print(f"  Running experiment {iteration}/{iterations} with payload_size={payload_size}, gates={num_gates}")
                
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
                        experiment_type="fixed_payload_experiments",
                        payload_size=payload_size,
                        num_gates=num_gates
                    )
                    
                    if ibm_result["status"] == "completed":
                        result_data.update({
                            "counts": self._serialize_dict(ibm_result["counts"]),
                            "success_rate": ibm_result["counts"].get('0' * payload_size, 0) / sum(ibm_result["counts"].values()),
                            "ibm_job_id": ibm_result.get("job_id"),
                            "ibm_backend": ibm_result.get("backend"),
                            "iteration": iteration
                        })
                    elif ibm_result["status"] == "error":
                        # Fallback to simulation if IBM execution failed
                        sim_result = validator.run_simulation(show_histogram=show_histogram)
                        result_data.update({
                            "status": "completed",
                            "counts": self._serialize_dict(sim_result["results_metrics"]["counts"]),
                            "success_rate": sim_result["results_metrics"]["success_rate"],
                            "ibm_error": ibm_result.get("error"),
                            "iteration": iteration
                        })
                else:
                    sim_result = validator.run_simulation(show_histogram=show_histogram)
                    result_data = self._prepare_result_data(
                        validator=validator,
                        status="completed",
                        execution_type="simulation",
                        experiment_type="fixed_payload_experiments",
                        payload_size=payload_size,
                        num_gates=num_gates,
                        counts=sim_result["results_metrics"]["counts"],
                        success_rate=sim_result["results_metrics"]["success_rate"]
                    )
                    result_data["iteration"] = iteration
                
                # Append to DataFrame
                self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
                print(f"  Experiment completed with status: {result_data['status']}, circuit depth: {validator.circuit.depth()}")
                
                # Save after each iteration to ensure data is not lost
                self.export_to_csv(output_file)
        
        return self.results_df
