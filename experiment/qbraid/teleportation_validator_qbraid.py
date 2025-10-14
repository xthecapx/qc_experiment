from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from IPython.display import display
import numpy as np
import random
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
# from results import results_1_4_2000_2005
from qiskit.providers.fake_provider import GenericBackendV2
from enum import Enum
from qbraid.transpiler import transpile as qbraid_transpile
from qbraid.runtime import QbraidProvider, load_job
from qbraid.runtime import JobLoaderError
try:
    from qbraid.runtime.aws import BraketProvider as QbraidBraketProvider
except ImportError:
    QbraidBraketProvider = None
try:
    from qiskit_braket_provider import BraketProvider as QiskitBraketProvider
except ImportError:
    QiskitBraketProvider = None
# from qbraid import circuit_wrapper

class QbraidDevice(Enum):
    IONQ = "aws_ionq"
    QIR = "qbraid_qir_simulator"
    LUCY = "aws_oqc_lucy"
    RIGETTI = "rigetti_ankaa_3"
    IBM_SANTIAGO = "ibm_q_santiago"
    IBM_SIMULATOR = "ibm_simulator"

# Load environment variables
load_dotenv()




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
        # if self.use_barriers:
        #     self.circuit.barrier()

        # Alice's operations on her qubits
        self.circuit.cx(self.message_qubit, self.alice_entangled)
        self.circuit.h(self.message_qubit)
        if self.use_barriers:
            self.circuit.barrier()

        # Bell measurement and classical communication
        self.circuit.cx(self.alice_entangled, self.bob_entangled)
        self.circuit.cz(self.message_qubit, self.bob_entangled)
        # if self.use_barriers:
        #     self.circuit.barrier()

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
        if self.use_barriers and not self.save_statevector:
            circuit.barrier()
    
        circuit = circuit.compose(
            self.protocol.circuit, 
            qubits=range(self.payload_size, self.payload_size + 3)
        )
        if self.use_barriers and not self.save_statevector:
            circuit.barrier()
        
        if self.save_statevector:
            circuit.save_statevector(label='before_validation')
    
        self._create_validation(circuit)
        if self.save_statevector:
            circuit.save_statevector(label='after_validation')
        if self.use_barriers and not self.save_statevector:
            circuit.barrier()
    
        circuit.add_register(self.result)
        circuit.measure(self.auxiliary_qubits, self.result)
        # circuit.measure_all()
    
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
            "num_ancillas": getattr(self.circuit, 'num_ancillas', 0),
            "num_parameters": getattr(self.circuit, 'num_parameters', 0),
            "has_calibrations": bool(getattr(self.circuit, 'calibrations', None)),
            "has_layout": bool(getattr(self.circuit, 'layout', None)),
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
        
    def run_qbraid(self):
        provider = QbraidProvider()
        # print(provider.get_devices())
        qbraid_device = provider.get_device(device_id=QbraidDevice.RIGETTI.value)
        print(qbraid_device)
        # display(self.draw())
        
        try:
            print(f"Original circuit depth: {self.circuit.depth()}")
            print(f"Original circuit width: {self.circuit.width()}")
            print(f"Device capabilities: {qbraid_device}")
            
            transpiled_circuit = qbraid_transpile(self.circuit, 'braket')
            print(f"Transpiled circuit type: {type(transpiled_circuit)}")
            # print(transpiled_circuit)
            
            # Check if device is available
            print(f"Device status: {qbraid_device.status()}")
            
            job = qbraid_device.run(transpiled_circuit, shots=10)
            
            # Immediately return job info for CSV storage
            print(f"Job submitted with ID: {job.id}")
            job_info = {
                "status": "submitted",
                "job_id": job.id,
                "device": qbraid_device.id,
                "job": job
            }
            
            # Wait for job completion
            print("Waiting for job to complete...")
            
            import time
            start_time = time.time()
            timeout = 300  # 5 minutes timeout
            
            while time.time() - start_time < timeout:
                status = job.status()
                print(f"Job status: {status}")
                
                # Convert enum to string for easier comparison
                status_str = str(status).split('.')[-1] if hasattr(status, 'name') else str(status)
                
                if status_str in ['COMPLETED', 'DONE']:
                    result = job.result()
                    counts = result.data.get_counts()
                    print(f"Job completed! Results: {counts}")
                    # Update job_info with completion data
                    job_info.update({
                        "status": "completed",
                        "counts": counts,
                        "transpiled_circuit": transpiled_circuit
                    })
                    return job_info
                elif status_str in ['FAILED', 'CANCELLED', 'CANCELED']:
                    print(f"Job failed with status: {status}")
                    # Try to get error details if available
                    error_msg = f"Job {status_str.lower()}"
                    try:
                        if hasattr(job, 'metadata') and job.metadata():
                            error_msg += f": {job.metadata()}"
                    except:
                        pass
                    
                    # Update job_info with failure data
                    job_info.update({
                        "status": "failed",
                        "error": error_msg
                    })
                    return job_info
                elif status_str in ['QUEUED', 'RUNNING', 'INITIALIZING']:
                    # Continue waiting for these status values
                    pass
                else:
                    print(f"Unknown status: {status} ({status_str})")
                
                time.sleep(5)  # Wait 5 seconds before checking again
            
            # Timeout case - still return job_info with job_id
            print("Job timed out after 5 minutes")
            job_info.update({
                "status": "timeout"
            })
            return job_info
        except Exception as e:
            print(f"Error running on qBraid: {str(e)}")
            counts = AerSimulator().run(self.circuit).result().get_counts()
            success_rate = counts.get('0' * self.circuit.num_qubits, 0) / sum(counts.values()) if counts else 0
            
            # Return comprehensive metrics matching expected format
            return {
                "status": "error",
                "error": str(e),
                "counts": counts,
                "success_rate": success_rate,
                "circuit_depth": self.circuit.depth(),
                "circuit_width": self.circuit.width(),
                "circuit_size": self.circuit.size(),
                "circuit_count_ops": self.circuit.count_ops(),
                "num_qubits": self.circuit.num_qubits,
                "num_clbits": self.circuit.num_clbits,
                "num_ancillas": getattr(self.circuit, 'num_ancillas', 0),
                "num_parameters": getattr(self.circuit, 'num_parameters', 0),
                "has_calibrations": bool(getattr(self.circuit, 'calibrations', None)),
                "has_layout": bool(getattr(self.circuit, 'layout', None)),
                "device": "simulator_fallback"
            }

    def retrieve_aws_job(self, job_id: str, region: str = "us-west-1"):
        """
        Retrieve results from a previously submitted AWS Braket job.
        
        Args:
            job_id (str): AWS Braket job ARN
            region (str): AWS region (default: "us-west-1")
        
        Returns:
            dict: Job results and information
        """
        if QiskitBraketProvider is None:
            return {"status": "error", "error": "qiskit-braket-provider not available"}
        
        try:
            import os
            
            # Set AWS region if not already set
            if 'AWS_DEFAULT_REGION' not in os.environ:
                os.environ['AWS_DEFAULT_REGION'] = region
            
            print(f"Retrieving AWS job: {job_id}")
            provider = QiskitBraketProvider()

            backend = provider.get_backend("Ankaa-3")
            if not backend:
                return {"status": "error", "error": "No AWS backends available"}
            
            job = backend.retrieve_job(job_id)
            
            status = job.status()
            print(f"Job status: {status}")

            try:
                # Get the AWS result and extract counts directly
                aws_result = job._tasks[0].result()
                measured_entry = aws_result.entries[0].entries[0]
                raw_counts = dict(measured_entry.counts)  # Convert Counter to dict
                
                # Convert from big-endian to little-endian
                counts = {k[::-1]: v for k, v in raw_counts.items()}
                print('counts', counts)
                
                # Determine the actual number of measured qubits from the counts
                if counts:
                    measured_qubits = len(list(counts.keys())[0])  # Get length of first key
                    success_pattern = '0' * measured_qubits  # All zeros for success
                else:
                    success_pattern = '0' * self.payload_size  # Fallback to payload_size
                
                success_rate = counts.get(success_pattern, 0) / sum(counts.values()) if counts else 0
                print(f'success_pattern: {success_pattern}')
                print('success_rate', success_rate)

                return {
                    "status": "completed",
                    "job_id": job_id,
                    "counts": str(counts),
                    "success_rate": success_rate
                }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "job_id": job_id,
                    "error": f"Error retrieving results: {str(e)}"
                }

                
        except Exception as e:
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            }
    
    def complete_aws_results_from_csv(self, csv_file_path: str, output_csv_path: str = None, region: str = "us-west-1"):
        """
        Complete AWS job results in CSV by retrieving results for submitted jobs.
        
        Args:
            csv_file_path (str): Path to the CSV file containing job IDs
            output_csv_path (str): Path to save the updated CSV (optional, defaults to input path)
            region (str): AWS region (default: "us-west-1")
        
        Returns:
            dict: Summary of processing results
        """
        if QiskitBraketProvider is None:
            print("❌ qiskit-braket-provider not available")
            return {"status": "error", "error": "qiskit-braket-provider not available"}
        
        try:
            import pandas as pd
            import os
            
            # Read the CSV file
            print(f"📖 Reading CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            if 'job_id' not in df.columns:
                return {"status": "error", "error": "CSV file must contain 'job_id' column"}
            
            # Filter rows that have job_id but no results yet
            pending_jobs = df[
                (df['job_id'].notna()) & 
                (df['job_id'] != '') & 
                (df['job_id'] != 'None') &
                (df['status'].isin(['submitted', 'timeout']) | df['counts'].isna())
            ].copy()
            
            print(f"🔍 Found {len(pending_jobs)} jobs to process")
            
            if len(pending_jobs) == 0:
                print("✅ No pending jobs found to process")
                return {"status": "completed", "processed": 0, "completed": 0, "errors": 0}
            
            # Initialize counters
            processed = 0
            completed = 0
            errors = 0
            
            # Process each pending job
            for idx, row in pending_jobs.iterrows():
                job_id = row['job_id']
                print(f"\n🔄 Processing job {processed + 1}/{len(pending_jobs)}: {job_id}")
                
                try:
                    # Create a temporary validator with the same payload_size for success_rate calculation
                    temp_validator = TeleportationValidator(
                        payload_size=int(row['payload_size']) if pd.notna(row['payload_size']) else 1,
                        use_barriers=False
                    )
                    
                    # Retrieve job results
                    result = temp_validator.retrieve_aws_job(job_id, region=region)
                    
                    # Update the DataFrame with results
                    if result['status'] == 'completed':
                        df.loc[idx, 'status'] = 'completed'
                        df.loc[idx, 'counts'] = result['counts']
                        df.loc[idx, 'success_rate'] = result['success_rate']
                        completed += 1
                        print(f"✅ Job completed - Success rate: {result['success_rate']:.3f}")
                    else:
                        df.loc[idx, 'status'] = result['status']
                        if 'error' in result:
                            df.loc[idx, 'error'] = result.get('error', '')
                        print(f"⚠️  Job status: {result['status']}")
                    
                    processed += 1
                    
                except Exception as e:
                    print(f"❌ Error processing job {job_id}: {str(e)}")
                    df.loc[idx, 'status'] = 'error'
                    df.loc[idx, 'error'] = f"Retrieval error: {str(e)}"
                    errors += 1
                    processed += 1
            
            # Save the updated CSV
            output_path = output_csv_path or csv_file_path
            print(f"\n💾 Saving updated results to: {output_path}")
            df.to_csv(output_path, index=False)
            
            # Print summary
            print(f"\n📊 Processing Summary:")
            print(f"   • Total processed: {processed}")
            print(f"   • Completed jobs: {completed}")
            print(f"   • Errors: {errors}")
            print(f"   • Still pending: {processed - completed - errors}")
            
            return {
                "status": "completed",
                "processed": processed,
                "completed": completed,
                "errors": errors,
                "output_file": output_path
            }
            
        except Exception as e:
            print(f"❌ Error processing CSV: {str(e)}")
            return {
                "status": "error",
                "error": f"CSV processing error: {str(e)}"
            }

    def run_aws(self, device_id: str = "Ankaa-3", shots: int = 10, region: str = "us-west-1"):
        """
        Run the quantum circuit on AWS Braket using the qiskit-braket-provider.
        
        Args:
            device_id (str): AWS Braket device ID (default: "Ankaa-3")
            shots (int): Number of shots to run (default: 10)
            region (str): AWS region (default: "us-west-1")
        
        Returns:
            dict: Job information and results
        """
        if QiskitBraketProvider is None:
            print("❌ qiskit-braket-provider not available or incompatible.")
            print("This is likely due to a version compatibility issue between Qiskit and qiskit-braket-provider.")
            print("Try running: !pip install --upgrade qiskit-braket-provider")
            print("Or check Qiskit version compatibility.")
            print("Falling back to simulation...")
            
            # Fallback to simulation
            print("🔄 Falling back to simulation...")
            sim_result = self.run_simulation(shots=shots)
            
            # Convert simulation result to CSV format
            return {
                "job_id": None,
                "status": "simulation_fallback",
                "circuit_depth": self.circuit.depth(),
                "circuit_width": self.circuit.width(),
                "circuit_size": len(self.circuit.data),
                "circuit_count_ops": str(dict(self.circuit.count_ops())),
                "num_qubits": self.circuit.num_qubits,
                "num_clbits": self.circuit.num_clbits,
                "num_ancillas": getattr(self.circuit, 'num_ancillas', 0),
                "num_parameters": self.circuit.num_parameters,
                "has_calibrations": bool(getattr(self.circuit, 'calibrations', None)),
                "has_layout": bool(getattr(self.circuit, 'layout', None)),
                "payload_size": self.payload_size,
                "num_gates": self.gates,
                "execution_type": "simulation",
                "experiment_type": "teleportation",
                "counts": str(sim_result.get("counts", {})),
                "success_rate": sim_result.get("success_rate", 0),
                "device": "simulator_fallback",
                "shots": shots
            }
        
        try:
            import os
            
            # Set AWS region if not already set
            if 'AWS_DEFAULT_REGION' not in os.environ:
                os.environ['AWS_DEFAULT_REGION'] = region
                print(f"Setting AWS region to: {region}")
            
            print(f"Setting up AWS Braket provider...")
            provider = QiskitBraketProvider()
            
            # Get the specified device
            print(f"Getting device: {device_id}")
            aws_device = provider.get_backend(device_id)
            print(f"Device obtained: {aws_device}")
            
            print(f"Original circuit depth: {self.circuit.depth()}")
            print(f"Original circuit width: {self.circuit.width()}")
            print(f"Device capabilities: {aws_device}")
            
            # Create a copy of the circuit and remove barriers for AWS compatibility
            circuit_copy = self.circuit.copy()
            circuit_copy.remove_final_measurements()
            circuit_copy.measure_all()
            
            # Remove barriers from the circuit copy
            from qiskit.circuit.library import Barrier
            circuit_without_barriers = circuit_copy.copy()
            circuit_without_barriers.data = [
                (gate, qubits, clbits) for gate, qubits, clbits in circuit_copy.data 
                if not isinstance(gate, Barrier)
            ]
            
            # Submit the job (no waiting)
            print(f"🚀 Submitting circuit to AWS Braket device: {device_id}")
            print(f"📊 Shots: {shots}")
            job = aws_device.run(circuit_without_barriers, shots=shots)
            
            print(f"✅ Job submitted successfully!")
            print(f"🆔 Job ID: {job.job_id()}")
            
            # Return job information for CSV (without waiting for execution)
            return {
                "job_id": job.job_id(),
                "status": "submitted",
                "circuit_depth": self.circuit.depth(),
                "circuit_width": self.circuit.width(),
                "circuit_size": len(self.circuit.data),
                "circuit_count_ops": str(dict(self.circuit.count_ops())),
                "num_qubits": self.circuit.num_qubits,
                "num_clbits": self.circuit.num_clbits,
                "num_ancillas": getattr(self.circuit, 'num_ancillas', 0),
                "num_parameters": self.circuit.num_parameters,
                "has_calibrations": bool(getattr(self.circuit, 'calibrations', None)),
                "has_layout": bool(getattr(self.circuit, 'layout', None)),
                "payload_size": self.payload_size,
                "num_gates": self.gates,
                "execution_type": "aws_braket",
                "experiment_type": "teleportation",
                "counts": None,  # Will be populated by retrieve_aws_job
                "success_rate": None,  # Will be populated by retrieve_aws_job
                "device": device_id,
                "shots": shots
            }
            
        except Exception as e:
            print(f"❌ Error running on AWS Braket: {e}")
            return {
                "job_id": None,
                "status": "error",
                "circuit_depth": self.circuit.depth(),
                "circuit_width": self.circuit.width(),
                "circuit_size": len(self.circuit.data),
                "circuit_count_ops": str(dict(self.circuit.count_ops())),
                "num_qubits": self.circuit.num_qubits,
                "num_clbits": self.circuit.num_clbits,
                "num_ancillas": getattr(self.circuit, 'num_ancillas', 0),
                "num_parameters": self.circuit.num_parameters,
                "has_calibrations": bool(getattr(self.circuit, 'calibrations', None)),
                "has_layout": bool(getattr(self.circuit, 'layout', None)),
                "payload_size": self.payload_size,
                "num_gates": self.gates,
                "execution_type": "aws_braket",
                "experiment_type": "teleportation",
                "counts": None,
                "success_rate": None,
                "error": f"AWS Job submission error: {str(e)}",
                "device": device_id,
                "shots": shots
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
                           payload_size, num_gates, counts=None, success_rate=None, job_id=None):
        """Helper method to prepare result data with proper serialization"""
        result_data = {
            "status": status,
            "circuit_depth": validator.circuit.depth(),
            "circuit_width": validator.circuit.width(),
            "circuit_size": validator.circuit.size(),
            "circuit_count_ops": self._serialize_dict(validator.circuit.count_ops()),
            "num_qubits": validator.circuit.num_qubits,
            "num_clbits": validator.circuit.num_clbits,
            "num_ancillas": getattr(validator.circuit, 'num_ancillas', 0),
            "num_parameters": getattr(validator.circuit, 'num_parameters', 0),
            "has_calibrations": bool(getattr(validator.circuit, 'calibrations', None)),
            "has_layout": bool(getattr(validator.circuit, 'layout', None)),
            "payload_size": payload_size,
            "num_gates": num_gates,
            "execution_type": execution_type,
            "experiment_type": experiment_type
        }
        
        # Add job_id if provided
        if job_id is not None:
            result_data["job_id"] = job_id
        
        if counts is not None:
            result_data["counts"] = self._serialize_dict(counts)
        if success_rate is not None:
            result_data["success_rate"] = success_rate
        
        return result_data

    def run_payload_size_gates_correlation(self, start: int = 1, end: int = 10, 
                                         use_qbraid: bool = False, show_circuit: bool = False):
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_payload_correlation_{timestamp}.csv'
        
        for size in range(start, end + 1):
            print(f"\nRunning experiment with payload_size={size} and gates={size}")
            
            validator = TeleportationValidator(
                payload_size=size,
                gates=size,
                use_barriers=True
            )
            
            if show_circuit:
                display(validator.draw())
            
            # Determine execution type and run experiment
            if use_qbraid:
                try:
                    qbraid_result = validator.run_qbraid()
                    if qbraid_result["status"] == "completed":
                        execution_type = "qbraid"
                        result_data = self._prepare_result_data(
                            validator=validator,
                            status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="payload_size_gates_correlation",
                            payload_size=size,
                            num_gates=size,
                            counts=qbraid_result["counts"],
                            success_rate=qbraid_result["counts"].get('0' * size, 0) / sum(qbraid_result["counts"].values()),
                            job_id=qbraid_result["job_id"]  # Always store job_id
                        )
                    elif qbraid_result["status"] in ["submitted", "timeout", "failed"]:
                        # Job was submitted but didn't complete - still store job_id for later retrieval
                        execution_type = "qbraid"
                        print(f"Job {qbraid_result['status']} - storing job_id for later retrieval: {qbraid_result['job_id']}")
                        # Use simulation results as fallback but mark as qbraid with job_id
                        counts = AerSimulator().run(validator.circuit).result().get_counts()
                        result_data = self._prepare_result_data(
                            validator=validator,
                            status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="payload_size_gates_correlation",
                            payload_size=size,
                            num_gates=size,
                            counts=counts,
                            success_rate=counts.get('0' * size, 0) / sum(counts.values()),
                            job_id=qbraid_result["job_id"]  # Store job_id for later API retrieval
                        )
                    elif qbraid_result["status"] == "error":
                        # Job submission failed - use simulation results but store error info
                        execution_type = "simulation"  # Mark as simulation since no qBraid job was created
                        print(f"Job submission error: {qbraid_result.get('error', 'Unknown error')}")
                        result_data = self._prepare_result_data(
                            validator=validator,
                            status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="payload_size_gates_correlation",
                            payload_size=size,
                            num_gates=size,
                            counts=qbraid_result["counts"],
                            success_rate=qbraid_result["success_rate"]
                        )
                        result_data["error"] = qbraid_result.get("error")
                    else:
                        # Fallback to simulation if qBraid execution failed
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
                except Exception as e:
                    print(f"qBraid execution failed: {e}, falling back to simulation")
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
                                      use_qbraid: bool = False, show_circuit: bool = False):
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_fixed_payload_{payload_size}_{timestamp}.csv'
        
        for num_gates in range(start_gates, end_gates + 1):
            print(f"\nRunning experiment with payload_size={payload_size} and gates={num_gates}")
            
            validator = TeleportationValidator(
                payload_size=payload_size,
                gates=num_gates,
                use_barriers=True
            )
            
            if show_circuit:
                display(validator.draw())
            
            # Determine execution type and run experiment
            if use_qbraid:
                try:
                    qbraid_result = validator.run_qbraid()
                    if qbraid_result["status"] == "completed":
                        execution_type = "qbraid"
                        result_data = self._prepare_result_data(
                            validator=validator,
                            status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="fixed_payload_varying_gates",
                            payload_size=payload_size,
                            num_gates=num_gates,
                            counts=qbraid_result["counts"],
                            success_rate=qbraid_result["counts"].get('0' * payload_size, 0) / sum(qbraid_result["counts"].values()),
                            job_id=qbraid_result["job_id"]
                        )
                    elif qbraid_result["status"] in ["submitted", "timeout", "failed"]:
                        execution_type = "qbraid"
                        print(f"Job {qbraid_result['status']} - storing job_id for later retrieval: {qbraid_result['job_id']}")
                        counts = AerSimulator().run(validator.circuit).result().get_counts()
                        result_data = self._prepare_result_data(
                            validator=validator,
                            status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="fixed_payload_varying_gates",
                            payload_size=payload_size,
                            num_gates=num_gates,
                            counts=counts,
                            success_rate=counts.get('0' * payload_size, 0) / sum(counts.values()),
                            job_id=qbraid_result["job_id"]
                        )
                    else:
                        # Fallback to simulation if qBraid execution failed
                        sim_result = validator.run_simulation()
                        result_data = self._prepare_result_data(
                            validator=validator,
                            status="completed",
                            execution_type="simulation",
                            experiment_type="fixed_payload_varying_gates",
                            payload_size=payload_size,
                            num_gates=num_gates,
                            counts=sim_result["results_metrics"]["counts"],
                            success_rate=sim_result["results_metrics"]["success_rate"]
                        )
                except Exception as e:
                    print(f"qBraid execution failed: {e}, falling back to simulation")
                    sim_result = validator.run_simulation()
                    result_data = self._prepare_result_data(
                        validator=validator,
                        status="completed",
                        execution_type="simulation",
                        experiment_type="fixed_payload_varying_gates",
                        payload_size=payload_size,
                        num_gates=num_gates,
                        counts=sim_result["results_metrics"]["counts"],
                        success_rate=sim_result["results_metrics"]["success_rate"]
                    )
            else:
                sim_result = validator.run_simulation()
                result_data = self._prepare_result_data(
                    validator=validator,
                    status="completed",
                    execution_type="simulation",
                    experiment_type="fixed_payload_varying_gates",
                    payload_size=payload_size,
                    num_gates=num_gates,
                    counts=sim_result["results_metrics"]["counts"],
                    success_rate=sim_result["results_metrics"]["success_rate"]
                )
            
            # Append to DataFrame
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_data])], ignore_index=True)
            print(f"Experiment {num_gates}/{end_gates} completed with status: {result_data['status']}")
            
            # Save after each iteration to ensure data is not lost
            self.export_to_csv(output_file)
        
        return self.results_df

    def run_dynamic_payload_gates(self, payload_range: tuple, gates_range: tuple,
                                use_qbraid: bool = False, use_aws: bool = False, 
                                aws_device_id: str = "Ankaa-3", show_circuit: bool = False):
        """
        Runs experiments with custom ranges for both payload size and number of gates.
        
        Args:
            payload_range: tuple of (min_payload, max_payload)
            gates_range: tuple of (min_gates, max_gates)
            use_qbraid: bool = False - Whether to run on qBraid quantum hardware
            use_aws: bool = False - Whether to run on AWS Braket quantum hardware
            aws_device_id: str = "Ankaa-3" - AWS Braket device ID to use
            show_circuit: bool = False - Whether to display circuit diagrams
        """
        start_payload, end_payload = payload_range
        start_gates, end_gates = gates_range
        
        # Create output filename with timestamp for this experiment run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'experiment_results_dynamic_{start_payload}-{end_payload}_{start_gates}-{end_gates}_{timestamp}.csv'
        
        for payload_size in range(start_payload, end_payload + 1):
            for num_gates in range(start_gates, end_gates + 1):
                print(f"\nRunning experiment with payload_size={payload_size} and gates={num_gates}")
                
                validator = TeleportationValidator(
                    payload_size=payload_size,
                    gates=num_gates,
                    use_barriers=False
                )
                
                if show_circuit:
                    display(validator.draw())
                
                # Determine execution type and run experiment
                if use_aws:
                    # Check if AWS is available, otherwise suggest qBraid alternative
                    if QiskitBraketProvider is None:
                        print(f"⚠️  AWS Braket provider not available due to dependency conflicts.")
                        print(f"💡 Suggestion: Use qBraid instead - it can access AWS devices including {aws_device_id}")
                        print(f"   Set use_qbraid=True instead of use_aws=True")
                        print(f"   Falling back to simulation for this run...")
                        
                        # Fallback to simulation with helpful message
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
                        result_data["note"] = f"AWS unavailable - use qBraid for {aws_device_id} access"
                    else:
                        try:
                            aws_result = validator.run_aws(device_id=aws_device_id, region="us-west-1")
                            if aws_result["status"] == "completed":
                                execution_type = "aws"
                                result_data = self._prepare_result_data(
                                    validator=validator,
                                    status=aws_result["status"],
                                    execution_type=execution_type,
                                    experiment_type="dynamic_payload_gates",
                                    payload_size=payload_size,
                                    num_gates=num_gates,
                                    counts=aws_result["counts"],
                                    success_rate=aws_result["success_rate"],
                                    job_id=aws_result["job_id"]
                                )
                            elif aws_result["status"] in ["submitted", "timeout", "failed"]:
                                execution_type = "aws"
                                print(f"AWS Job {aws_result['status']} - storing job_id for later retrieval: {aws_result['job_id']}")
                                counts = AerSimulator().run(validator.circuit).result().get_counts()
                                result_data = self._prepare_result_data(
                                    validator=validator,
                                    status=aws_result["status"],
                                    execution_type=execution_type,
                                    experiment_type="dynamic_payload_gates",
                                    payload_size=payload_size,
                                    num_gates=num_gates,
                                    counts=counts,
                                    success_rate=counts.get('0' * payload_size, 0) / sum(counts.values()),
                                    job_id=aws_result["job_id"]
                                )
                            elif aws_result["status"] == "error":
                                # Job submission failed - use simulation results but store error info
                                execution_type = "simulation"  # Mark as simulation since no AWS job was created
                                print(f"AWS Job submission error: {aws_result.get('error', 'Unknown error')}")
                                result_data = self._prepare_result_data(
                                    validator=validator,
                                    status=aws_result["status"],
                                    execution_type=execution_type,
                                    experiment_type="dynamic_payload_gates",
                                    payload_size=payload_size,
                                    num_gates=num_gates,
                                    counts=aws_result["counts"],
                                    success_rate=aws_result["success_rate"]
                                )
                                result_data["error"] = aws_result.get("error")
                            else:
                                # Fallback to simulation if AWS execution failed
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
                        except Exception as e:
                            print(f"AWS execution failed: {e}, falling back to simulation")
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
                elif use_qbraid:
                    try:
                        qbraid_result = validator.run_qbraid()
                        if qbraid_result["status"] == "completed":
                            execution_type = "qbraid"
                            result_data = self._prepare_result_data(
                            validator=validator,
                                status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="dynamic_payload_gates",
                            payload_size=payload_size,
                                num_gates=num_gates,
                                counts=qbraid_result["counts"],
                                success_rate=qbraid_result["counts"].get('0' * payload_size, 0) / sum(qbraid_result["counts"].values()),
                                job_id=qbraid_result["job_id"]
                            )
                        elif qbraid_result["status"] in ["submitted", "timeout", "failed"]:
                            execution_type = "qbraid"
                            print(f"Job {qbraid_result['status']} - storing job_id for later retrieval: {qbraid_result['job_id']}")
                            counts = AerSimulator().run(validator.circuit).result().get_counts()
                            result_data = self._prepare_result_data(
                            validator=validator,
                                status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="dynamic_payload_gates",
                            payload_size=payload_size,
                                num_gates=num_gates,
                                counts=counts,
                                success_rate=counts.get('0' * payload_size, 0) / sum(counts.values()),
                                job_id=qbraid_result["job_id"]
                            )
                        else:
                            # Fallback to simulation if qBraid execution failed
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
                    except Exception as e:
                        print(f"qBraid execution failed: {e}, falling back to simulation")
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
                                      use_qbraid: bool = False, show_circuit: bool = False, show_histogram: bool = False):
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
                
                validator = TeleportationValidator(
                    payload_size=payload_size,
                    gates=gates,
                    use_barriers=True
                )
                
                if show_circuit:
                    display(validator.draw())
                
                # Determine execution type and run experiment
                if use_qbraid:
                    try:
                        qbraid_result = validator.run_qbraid()
                        if qbraid_result["status"] == "completed":
                            execution_type = "qbraid"
                            result_data = self._prepare_result_data(
                            validator=validator,
                                    status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="controlled_depth_experiment",
                            payload_size=payload_size,
                                num_gates=gates,
                                counts=qbraid_result["counts"],
                                success_rate=qbraid_result["counts"].get('0' * payload_size, 0) / sum(qbraid_result["counts"].values()),
                                job_id=qbraid_result["job_id"]
                            )
                        elif qbraid_result["status"] in ["submitted", "timeout", "failed"]:
                            execution_type = "qbraid"
                            print(f"Job {qbraid_result['status']} - storing job_id for later retrieval: {qbraid_result['job_id']}")
                            counts = AerSimulator().run(validator.circuit).result().get_counts()
                            result_data = self._prepare_result_data(
                            validator=validator,
                                    status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="controlled_depth_experiment",
                            payload_size=payload_size,
                                num_gates=gates,
                                counts=counts,
                                success_rate=counts.get('0' * payload_size, 0) / sum(counts.values()),
                                job_id=qbraid_result["job_id"]
                            )
                        else:
                            # Fallback to simulation if qBraid execution failed
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
                    except Exception as e:
                        print(f"qBraid execution failed: {e}, falling back to simulation")
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
                                  use_qbraid: bool = True, show_circuit: bool = False, show_histogram: bool = False, min_experiments_per_depth: int = 5):
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
                
                validator = TeleportationValidator(
                    payload_size=payload_size,
                    gates=gates,
                    use_barriers=True
                )
                
                if show_circuit:
                    display(validator.draw())
                
                # Determine execution type and run experiment
                if use_qbraid:
                    try:
                        qbraid_result = validator.run_qbraid()
                        if qbraid_result["status"] == "completed":
                            execution_type = "qbraid"
                            result_data = self._prepare_result_data(
                            validator=validator,
                                    status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="target_depth_experiment",
                            payload_size=payload_size,
                                    num_gates=gates,
                                    counts=qbraid_result["counts"],
                                    success_rate=qbraid_result["counts"].get('0' * payload_size, 0) / sum(qbraid_result["counts"].values()),
                                    job_id=qbraid_result["job_id"]
                                )
                            result_data["target_depth"] = depth
                        elif qbraid_result["status"] in ["submitted", "timeout", "failed"]:
                            execution_type = "qbraid"
                            print(f"Job {qbraid_result['status']} - storing job_id for later retrieval: {qbraid_result['job_id']}")
                            counts = AerSimulator().run(validator.circuit).result().get_counts()
                            result_data = self._prepare_result_data(
                            validator=validator,
                                    status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="target_depth_experiment",
                            payload_size=payload_size,
                                    num_gates=gates,
                                    counts=counts,
                                    success_rate=counts.get('0' * payload_size, 0) / sum(counts.values()),
                                    job_id=qbraid_result["job_id"]
                                )
                            result_data["target_depth"] = depth
                        elif qbraid_result["status"] == "error":
                            # Job submission failed - use simulation results but store error info
                            execution_type = "simulation"  # Mark as simulation since no qBraid job was created
                            print(f"Job submission error: {qbraid_result.get('error', 'Unknown error')}")
                            result_data = self._prepare_result_data(
                                validator=validator,
                                status=qbraid_result["status"],
                                execution_type=execution_type,
                                experiment_type="target_depth_experiment",
                                payload_size=payload_size,
                                num_gates=gates,
                                counts=qbraid_result["counts"],
                                success_rate=qbraid_result["success_rate"]
                            )
                            result_data["target_depth"] = depth
                            result_data["error"] = qbraid_result.get("error")
                        else:
                            # Fallback to simulation if qBraid execution failed
                            sim_result = validator.run_simulation(show_histogram=show_histogram)
                            result_data = self._prepare_result_data(
                                validator=validator,
                                status="completed",
                                execution_type="simulation",
                                experiment_type="target_depth_experiment",
                                payload_size=payload_size,
                                num_gates=gates,
                                counts=sim_result["results_metrics"]["counts"],
                                success_rate=sim_result["results_metrics"]["success_rate"],
                                job_id=None
                            )
                            result_data["target_depth"] = depth
                    except Exception as e:
                        print(f"qBraid execution failed: {e}, falling back to simulation")
                        sim_result = validator.run_simulation(show_histogram=show_histogram)
                        result_data = self._prepare_result_data(
                            validator=validator,
                            status="completed",
                            execution_type="simulation",
                            experiment_type="target_depth_experiment",
                            payload_size=payload_size,
                            num_gates=gates,
                            counts=sim_result["results_metrics"]["counts"],
                            success_rate=sim_result["results_metrics"]["success_rate"],
                            job_id=None
                        )
                        result_data["target_depth"] = depth
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
                        success_rate=sim_result["results_metrics"]["success_rate"],
                        job_id=None
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
        Update experiment results table with detailed qBraid job information using qBraid SDK.
        
        Args:
            input_csv (str): Path to the input CSV file with experiment results
            output_csv (str): Path to save the updated CSV file (defaults to input_csv + '_updated.csv')
        
        Returns:
            pd.DataFrame: Updated DataFrame with job information
        """
        # Setup
        if output_csv is None:
            output_csv = input_csv.replace('.csv', '_updated.csv')
        
        df = pd.read_csv(input_csv)
        print(f"Processing {len(df)} rows from CSV file: {input_csv}")
        
        # Add new columns
        self._add_job_info_columns(df)
        
        # Process qBraid jobs
        stats = {'updated': 0, 'errors': 0, 'no_job_id': 0}
        
        for idx, row in df.iterrows():
            # Process any row that has a job_id, regardless of execution_type
            job_id = self._extract_job_id(row, idx)
            
            if job_id:
                self._process_job(df, idx, job_id, stats)
            else:
                # Only count as no_job_id if execution_type was intended to be qbraid
                if row.get('execution_type') == 'qbraid':
                    self._handle_no_job_id(df, idx, stats)
        
        # Add data source classification
        df['data_source'] = df.apply(lambda row: 
            'quantum' if (pd.notna(row.get('job_id')) and 
                         pd.notna(row.get('job_status')) and
                         str(row.get('job_status')) not in ['no_job_id', 'sdk_error'] and
                         (pd.notna(row.get('measurement_counts_sdk')) or pd.notna(row.get('counts')))) 
            else 'simulation', axis=1)
        
        # Display DataFrame before saving
        print("\n" + "="*80)
        print("FINAL DATAFRAME PREVIEW")
        print("="*80)
        print(f"Shape: {df.shape}")
        print(f"Job-related columns: {[col for col in df.columns if any(x in col.lower() for x in ['job', 'vendor', 'provider', 'cost', 'shots'])]}")
        print("\nFirst 3 rows with key columns:")
        key_cols = ['job_id', 'qbraid_device_id', 'job_status', 'vendor', 'provider', 'cost', 'shots', 'measurement_counts_sdk']
        available_cols = [col for col in key_cols if col in df.columns]
        print(df[available_cols].head(3).to_string())
        
        # Save and summarize
        df.to_csv(output_csv, index=False)
        print(f"\nUpdated results saved to {output_csv}")
        self._print_update_summary(df, stats['updated'], stats['errors'], stats['no_job_id'])
        
        return df
    
    def _process_job(self, df, idx, job_id, stats):
        """Process a single job and extract its data."""
        provider = self._determine_provider(job_id)
        print(f"Loading job {job_id} using {provider} provider...")
        
        try:
            quantum_job = load_job(job_id, provider=provider)
            print(f"✓ Job loaded successfully")
            
            # Extract all job data
            self._extract_job_data(df, idx, quantum_job, job_id, provider)
            stats['updated'] += 1
            
        except Exception as e:
            print(f"✗ Error loading job: {e}")
            self._handle_job_error(df, idx, job_id, provider, e, stats)
    
    def _handle_job_error(self, df, idx, job_id, provider, error, stats):
        """Handle job loading errors."""
        error_msg = str(error)
        df.at[idx, 'qbraid_job_id'] = job_id
        df.at[idx, 'provider'] = provider
        
        if "device" in error_msg.lower() and 'rigetti_ankaa_3' in job_id:
            # Infer device info from job ID
            df.at[idx, 'qbraid_device_id'] = 'rigetti_ankaa_3'
            df.at[idx, 'vendor'] = 'rigetti'
            df.at[idx, 'job_status'] = 'device_error_but_inferred'
        else:
            df.at[idx, 'job_status'] = f'error: {type(error).__name__}'
        
        stats['errors'] += 1
    
    def _handle_no_job_id(self, df, idx, stats):
        """Handle rows with no job ID."""
        print(f"Row {idx}: No job_id found - likely simulation")
        df.at[idx, 'job_status'] = 'no_job_id'
        stats['no_job_id'] += 1
    
    def _add_job_info_columns(self, df):
        """Add new columns for job information with appropriate data types."""
        new_columns = {
            # String columns
            'qbraid_job_id': 'object', 'qbraid_device_id': 'object', 'job_status': 'object',
            'vendor': 'object', 'provider': 'object', 'measurement_counts_sdk': 'object',
            'job_metadata': 'object', 'job_tags': 'object', 'job_created_at': 'object', 'job_ended_at': 'object',
            # Numeric columns
            'cost': 'float64', 'job_execution_duration': 'float64',
            'shots': 'int64', 'queue_position': 'int64', 'queue_depth': 'int64',
            'circuit_num_qubits': 'int64', 'circuit_depth_sdk': 'int64'
        }
        
        for col, dtype in new_columns.items():
            if col not in df.columns:
                df[col] = pd.Series(dtype=dtype)
    
    def _extract_job_id(self, row, idx):
        """Extract job ID from row data."""
        if pd.notna(row.get('job_id')):
            job_id = row['job_id']
            print(f"Row {idx}: Processing job_id = {job_id}")
            return job_id
        elif pd.notna(row.get('qbraid_job_id')):
            job_id = row['qbraid_job_id']
            print(f"Row {idx}: Processing qbraid_job_id = {job_id}")
            return job_id
        return None
    
    def _determine_provider(self, job_id):
        """Determine provider based on job ID format."""
        return 'aws' if job_id.startswith('arn:aws:braket:') else 'qbraid'
    
    def _extract_job_data(self, df, idx, quantum_job, job_id, provider):
        """Extract comprehensive data from a loaded job."""
        # Basic info
        df.at[idx, 'qbraid_job_id'] = job_id
        df.at[idx, 'provider'] = provider
        
        # Extract metadata
        self._extract_metadata(df, idx, quantum_job)
        
        # Extract result data
        self._extract_result_data(df, idx, quantum_job)
        
        # Extract queue position
        try:
            queue_pos = quantum_job.queue_position()
            if queue_pos is not None:
                df.at[idx, 'queue_position'] = queue_pos
        except:
            pass
    
    def _extract_metadata(self, df, idx, quantum_job):
        """Extract metadata from job."""
        try:
            metadata = quantum_job.metadata()
            # print(f"  📊 Metadata keys available: {list(metadata.keys())}")
            
            # Device and vendor info
            if 'device_id' in metadata:
                device_id = metadata['device_id']
                df.at[idx, 'qbraid_device_id'] = device_id
                print(f"  ✓ Device ID: {device_id}")
                
                device_id_lower = device_id.lower()
                if 'rigetti' in device_id_lower:
                    df.at[idx, 'vendor'] = 'rigetti'
                    print(f"  ✓ Vendor: rigetti")
                elif 'ibm' in device_id_lower:
                    df.at[idx, 'vendor'] = 'ibm'
                    print(f"  ✓ Vendor: ibm")
            
            # Job details
            status = metadata.get('status', '')
            df.at[idx, 'job_status'] = str(status)
            print(f"  ✓ Status: {status}")
            
            shots = metadata.get('shots')
            if shots is not None:
                df.at[idx, 'shots'] = shots
                print(f"  ✓ Shots: {shots}")
            
            tags = metadata.get('tags', {})
            df.at[idx, 'job_tags'] = str(tags)
            df.at[idx, 'job_metadata'] = str(metadata)
            
            # Timing information
            timestamps = metadata.get('time_stamps', {})
            print(f"  📅 Timestamps available: {list(timestamps.keys()) if isinstance(timestamps, dict) else 'None'}")
            
            if isinstance(timestamps, dict):
                created_at = timestamps.get('createdAt')
                ended_at = timestamps.get('endedAt')
                duration = timestamps.get('executionDuration')
                
                if created_at:
                    df.at[idx, 'job_created_at'] = created_at
                    print(f"  ✓ Created: {created_at}")
                if ended_at:
                    df.at[idx, 'job_ended_at'] = ended_at
                    print(f"  ✓ Ended: {ended_at}")
                if duration:
                    df.at[idx, 'job_execution_duration'] = duration
                    print(f"  ✓ Duration: {duration}ms")
            
            queue_pos = metadata.get('queue_position')
            if queue_pos is not None:
                df.at[idx, 'queue_position'] = queue_pos
                print(f"  ✓ Queue position: {queue_pos}")
            
        except Exception as e:
            print(f"  ❌ Metadata extraction error: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_result_data(self, df, idx, quantum_job):
        """Extract result data from job."""
        try:
            result = quantum_job.result()
            # print(f"  📈 Result type: {type(result)}")
            # print(f"  📈 Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            # Measurement counts (using recommended approach)
            if hasattr(result, 'data') and hasattr(result.data, 'get_counts'):
                try:
                    counts = result.data.get_counts()
                    df.at[idx, 'measurement_counts_sdk'] = str(counts)
                    print(f"  ✓ Measurement counts (get_counts): {counts}")
                except Exception as count_err:
                    print(f"  ⚠️ get_counts() failed: {count_err}")
                    # Fallback to direct measurement_counts access
                    if hasattr(result.data, 'measurement_counts'):
                        counts = result.data.measurement_counts
                        df.at[idx, 'measurement_counts_sdk'] = str(counts)
                        print(f"  ✓ Measurement counts (fallback): {counts}")
            elif hasattr(result, 'data') and hasattr(result.data, 'measurement_counts'):
                counts = result.data.measurement_counts
                df.at[idx, 'measurement_counts_sdk'] = str(counts)
                print(f"  ✓ Measurement counts (direct): {counts}")
            
            # COMPREHENSIVE COST SEARCH
            cost_found = False
            
            # Check all possible cost locations
            for cost_attr in ['cost', 'Cost', 'credits', 'Credits']:
                if hasattr(result, cost_attr):
                    cost_value = getattr(result, cost_attr)
                    df.at[idx, 'cost'] = str(cost_value)
                    print(f"  💰 Cost found in result.{cost_attr}: {cost_value}")
                    cost_found = True
                    break
            
            # Check in result details/kwargs and string representation
            if not cost_found:
                # Try string parsing first (most reliable method we found)
                result_str = str(result)
                import re
                cost_patterns = [
                    r"cost=Credits\('([^']+)'\)",
                    r"cost=([0-9.]+)",
                    r"cost='([^']+)'",
                    r"cost=\"([^\"]+)\""
                ]
                for pattern in cost_patterns:
                    match = re.search(pattern, result_str)
                    if match:
                        cost_value = float(match.group(1))
                        df.at[idx, 'cost'] = cost_value
                        print(f"  💰 Cost: {cost_value}")
                        cost_found = True
                        break
                
                # Fallback: check private attributes
                if not cost_found and hasattr(result, '__dict__'):
                    result_dict = result.__dict__
                    for private_attr in ['_details', '_data']:
                        if private_attr in result_dict:
                            private_obj = result_dict[private_attr]
                            if hasattr(private_obj, '__dict__'):
                                private_dict = private_obj.__dict__
                                for key, value in private_dict.items():
                                    if 'cost' in key.lower() or 'credit' in key.lower():
                                        try:
                                            df.at[idx, 'cost'] = float(str(value))
                                            print(f"  💰 Cost: {value}")
                                            cost_found = True
                                            break
                                        except ValueError:
                                            pass
                            if cost_found:
                                break
            
            if not cost_found:
                print(f"  ❌ No cost attribute found in result")
            
            # COMPREHENSIVE CIRCUIT METADATA SEARCH
            metadata_found = False
            
            # Check result.metadata
            if hasattr(result, 'metadata') and result.metadata:
                result_meta = result.metadata
                print(f"  🔧 Result metadata type: {type(result_meta)}")
                print(f"  🔧 Result metadata attributes: {dir(result_meta)}")
                
                # Try different access methods
                for method in ['get', '__getitem__', '__dict__']:
                    try:
                        if method == 'get' and hasattr(result_meta, 'get'):
                            circuit_qubits = result_meta.get('circuitNumQubits')
                            circuit_depth = result_meta.get('circuitDepth')
                        elif method == '__getitem__':
                            circuit_qubits = result_meta['circuitNumQubits'] if 'circuitNumQubits' in result_meta else None
                            circuit_depth = result_meta['circuitDepth'] if 'circuitDepth' in result_meta else None
                        elif method == '__dict__' and hasattr(result_meta, '__dict__'):
                            meta_dict = result_meta.__dict__
                            circuit_qubits = meta_dict.get('circuitNumQubits') or meta_dict.get('circuit_num_qubits')
                            circuit_depth = meta_dict.get('circuitDepth') or meta_dict.get('circuit_depth')
                        
                        if circuit_qubits:
                            df.at[idx, 'circuit_num_qubits'] = circuit_qubits
                            print(f"  ✓ Circuit qubits ({method}): {circuit_qubits}")
                            metadata_found = True
                        if circuit_depth:
                            df.at[idx, 'circuit_depth_sdk'] = circuit_depth
                            print(f"  ✓ Circuit depth ({method}): {circuit_depth}")
                            metadata_found = True
                            
                        if metadata_found:
                            break
                    except Exception as meta_err:
                        print(f"  ⚠️ Method {method} failed: {meta_err}")
                        continue
            
            # Check if metadata is at the top level of result
            if not metadata_found:
                for attr in ['circuitNumQubits', 'circuit_num_qubits', 'circuitDepth', 'circuit_depth']:
                    if hasattr(result, attr):
                        value = getattr(result, attr)
                        if 'qubit' in attr.lower():
                            df.at[idx, 'circuit_num_qubits'] = value
                            print(f"  ✓ Circuit qubits (result.{attr}): {value}")
                        elif 'depth' in attr.lower():
                            df.at[idx, 'circuit_depth_sdk'] = value
                            print(f"  ✓ Circuit depth (result.{attr}): {value}")
                        metadata_found = True
            
            # Check in private attributes for circuit metadata
            if not metadata_found and hasattr(result, '__dict__'):
                result_dict = result.__dict__
                for private_attr in ['_details', '_data']:
                    if private_attr in result_dict:
                        private_obj = result_dict[private_attr]
                        print(f"  🔍 Searching for circuit metadata in {private_attr}: {type(private_obj)}")
                        
                        # Check if private_obj has circuit attributes
                        for attr in ['circuitNumQubits', 'circuit_num_qubits', 'circuitDepth', 'circuit_depth', 'metadata']:
                            if hasattr(private_obj, attr):
                                value = getattr(private_obj, attr)
                                print(f"  🔍 Found {private_attr}.{attr}: {value} (type: {type(value)})")
                                
                                if attr == 'metadata' and value:
                                    # Found metadata object, try to extract circuit info from it
                                    if hasattr(value, 'get'):
                                        circuit_qubits = value.get('circuitNumQubits')
                                        circuit_depth = value.get('circuitDepth')
                                    elif hasattr(value, '__dict__'):
                                        meta_dict = value.__dict__
                                        circuit_qubits = meta_dict.get('circuitNumQubits') or meta_dict.get('circuit_num_qubits')
                                        circuit_depth = meta_dict.get('circuitDepth') or meta_dict.get('circuit_depth')
                                    else:
                                        circuit_qubits = circuit_depth = None
                                    
                                    if circuit_qubits:
                                        df.at[idx, 'circuit_num_qubits'] = circuit_qubits
                                        print(f"  ✓ Circuit qubits from {private_attr}.metadata: {circuit_qubits}")
                                        metadata_found = True
                                    if circuit_depth:
                                        df.at[idx, 'circuit_depth_sdk'] = circuit_depth
                                        print(f"  ✓ Circuit depth from {private_attr}.metadata: {circuit_depth}")
                                        metadata_found = True
                                        
                                elif 'qubit' in attr.lower():
                                    df.at[idx, 'circuit_num_qubits'] = value
                                    print(f"  ✓ Circuit qubits from {private_attr}.{attr}: {value}")
                                    metadata_found = True
                                elif 'depth' in attr.lower():
                                    df.at[idx, 'circuit_depth_sdk'] = value
                                    print(f"  ✓ Circuit depth from {private_attr}.{attr}: {value}")
                                    metadata_found = True
                        
                        if metadata_found:
                            break
            
            # Try to parse circuit metadata from string representation (most reliable method)
            if not metadata_found:
                result_str = str(result)
                import re
                circuit_patterns = [
                    r"circuitNumQubits[:\s]*([0-9]+)",
                    r"circuitDepth[:\s]*([0-9]+)",
                    r"circuit_num_qubits[:\s]*([0-9]+)",
                    r"circuit_depth[:\s]*([0-9]+)"
                ]
                qubits_found = depth_found = False
                for pattern in circuit_patterns:
                    match = re.search(pattern, result_str)
                    if match:
                        value = int(match.group(1))
                        if 'qubit' in pattern.lower() and not qubits_found:
                            df.at[idx, 'circuit_num_qubits'] = value
                            print(f"  🔧 Circuit qubits: {value}")
                            qubits_found = True
                        elif 'depth' in pattern.lower() and not depth_found:
                            df.at[idx, 'circuit_depth_sdk'] = value
                            print(f"  🔧 Circuit depth: {value}")
                            depth_found = True
                        if qubits_found and depth_found:
                            metadata_found = True
                            break
            
            if not metadata_found:
                print(f"  ❌ No circuit metadata found")
            
            # Additional fields
            for attr in ['shots', 'device_id']:
                if hasattr(result, attr):
                    value = getattr(result, attr)
                    print(f"  📊 Result.{attr}: {value}")
                    
                    if attr == 'device_id' and pd.isna(df.at[idx, 'qbraid_device_id']):
                        df.at[idx, 'qbraid_device_id'] = value
                        print(f"  ✓ Updated device_id from result: {value}")
                    elif attr == 'shots' and pd.isna(df.at[idx, 'shots']):
                        df.at[idx, 'shots'] = value
                        print(f"  ✓ Updated shots from result: {value}")
                        
        except Exception as e:
            print(f"  ❌ Result extraction error: {e}")
            import traceback
            traceback.print_exc()

    def _print_update_summary(self, df, updated_count, error_count, no_job_id_count):
        """Print a summary of the SDK-based update results"""
        print("\n" + "="*60)
        print("SDK UPDATE SUMMARY REPORT")
        print("="*60)
        
        total_job_rows = len(df[pd.notna(df['job_id'])])
        quantum_data_rows = len(df[df['data_source'] == 'quantum'])
        simulation_data_rows = len(df[df['data_source'] == 'simulation'])
        
        print(f"Total rows with job_id processed: {total_job_rows}")
        print(f"Jobs successfully updated with SDK: {updated_count}")
        print(f"Jobs with SDK errors: {error_count}")
        print(f"Jobs without job_id: {no_job_id_count}")
        print(f"Rows with actual quantum data: {quantum_data_rows}")
        print(f"Rows with simulation data: {simulation_data_rows}")
        
        # Status breakdown
        if 'job_status' in df.columns:
            status_counts = df['job_status'].value_counts()
            print(f"\nJob Status Breakdown:")
            for status, count in status_counts.items():
                if pd.notna(status):
                    print(f"  {status}: {count}")
        
        # Data source breakdown
        print(f"\nData Source Breakdown:")
        print(f"  Quantum hardware results: {quantum_data_rows}")
        print(f"  Simulation fallback results: {simulation_data_rows}")
        
        # Cost information if available
        if 'cost' in df.columns:
            total_cost = df['cost'].sum()
            if total_cost > 0:
                print(f"\nTotal job cost: ${total_cost:.6f}")
        
        # Data verification summary
        if 'measurement_counts_sdk' in df.columns:
            sdk_data_rows = len(df[pd.notna(df['measurement_counts_sdk'])])
            print(f"\nData Verification:")
            print(f"  Jobs with SDK measurement data: {sdk_data_rows}")
            print(f"  Jobs with existing CSV data: {total_job_rows}")
        
        print("="*60)

    def create_table_from_results(self, results_list: list):
        """
        Creates a DataFrame from experiment results.
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

                    'counts': json.dumps(experiment.get('results_metrics', {}).get('counts', {})),
                    'success_rate': experiment.get('results_metrics', {}).get('success_rate', 0.0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        
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
                                  use_qbraid: bool = False, show_circuit: bool = False, show_histogram: bool = False):
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
            use_qbraid: Whether to run on qBraid quantum hardware (default: False)
            show_circuit: Whether to display the circuit diagram (default: False)
            show_histogram: Whether to display histograms of measurement results (default: False)
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
                
                validator = TeleportationValidator(
                    payload_size=payload_size,
                    gates=num_gates,
                    use_barriers=True
                )
                
                if show_circuit:
                    display(validator.draw())
                
                # Determine execution type and run experiment
                if use_qbraid:
                    try:
                        qbraid_result = validator.run_qbraid()
                        if qbraid_result["status"] == "completed":
                            execution_type = "qbraid"
                            result_data = self._prepare_result_data(
                            validator=validator,
                                status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="fixed_payload_experiments",
                            payload_size=payload_size,
                                num_gates=num_gates,
                                counts=qbraid_result["counts"],
                                success_rate=qbraid_result["counts"].get('0' * payload_size, 0) / sum(qbraid_result["counts"].values()),
                                job_id=qbraid_result["job_id"]
                            )
                            result_data["iteration"] = iteration
                        elif qbraid_result["status"] in ["submitted", "timeout", "failed"]:
                            execution_type = "qbraid"
                            print(f"Job {qbraid_result['status']} - storing job_id for later retrieval: {qbraid_result['job_id']}")
                            counts = AerSimulator().run(validator.circuit).result().get_counts()
                            result_data = self._prepare_result_data(
                            validator=validator,
                                status=qbraid_result["status"],
                            execution_type=execution_type,
                            experiment_type="fixed_payload_experiments",
                            payload_size=payload_size,
                                num_gates=num_gates,
                                counts=counts,
                                success_rate=counts.get('0' * payload_size, 0) / sum(counts.values()),
                                job_id=qbraid_result["job_id"]
                            )
                            result_data["iteration"] = iteration
                        else:
                            # Fallback to simulation if qBraid execution failed
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
                    except Exception as e:
                        print(f"qBraid execution failed: {e}, falling back to simulation")
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
