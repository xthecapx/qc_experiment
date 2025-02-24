from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qbraid.transpiler import transpile as qbraid_transpile
from qbraid.runtime import QbraidProvider
from enum import Enum
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from results import results_1_5_200_205

class   QbraidDevice(Enum):
    IONQ = "aws_ionq"
    QIR = "qbraid_qir_simulator"
    LUCY = "aws_oqc_lucy"
    RIGETTI = "aws_rigetti_aspen_m_1"
    IBM_SANTIAGO = "ibm_q_santiago"
    IBM_SIMULATOR = "ibm_simulator"

class GateFidelity:
    def __init__(self):
        self.qubit = QuantumRegister(1, "Q")
        self.ebit0 = QuantumRegister(1, "A")
        self.ebit1 = QuantumRegister(1, "B")
        self.payload_size = 1
        self.aux = QuantumRegister(self.payload_size, "R")
        
    def create_protocol(self):
        protocol = QuantumCircuit(self.qubit, self.ebit0, self.ebit1)
        
        # Prepare ebit used for teleportation
        protocol.h(self.ebit0)
        protocol.cx(self.ebit0, self.ebit1)
        protocol.barrier()
        
        # Alice's operations
        protocol.cx(self.qubit, self.ebit0)
        protocol.h(self.qubit)
        protocol.barrier()
        
        # Alice measures and sends classical bits to Bob
        protocol.cx(self.ebit0, self.ebit1)
        protocol.cz(self.qubit, self.ebit1)
        protocol.barrier()
        
        return protocol
    
    def create_circuit(self, pre_gate='s', post_gate='sdg'):
        # Create quantum circuit
        qc = QuantumCircuit(self.aux, self.qubit, self.ebit0, self.ebit1)
        
        # Entangle Q with R
        for qreg in self.aux:
            qc.h(qreg)
            qc.cx(qreg, self.qubit)
            getattr(qc, pre_gate)(qreg)
        qc.barrier()
        
        # Append protocol
        protocol = self.create_protocol()
        qc = qc.compose(protocol, qubits=range(self.payload_size, self.payload_size+3))
        qc.barrier()
        
        # Verification steps
        for qreg in reversed(self.aux):
            getattr(qc, post_gate)(qreg)
            qc.cx(qreg, self.ebit1)
            qc.h(qreg)
        
        qc.barrier()
        
        # Measurement
        result = ClassicalRegister(self.payload_size, "test_result")
        qc.add_register(result)
        qc.measure(self.aux, result)
        
        return qc
    
    def run_simulation(self, pre_gate='s', post_gate='sdg'):
        qc = self.create_circuit(pre_gate, post_gate)
        counts = AerSimulator().run(qc).result().get_counts()
        return counts, qc
    
    def run_qbraid(self, device: QbraidDevice = QbraidDevice.QIR, pre_gate='s', post_gate='sdg'):
        qc = self.create_circuit(pre_gate, post_gate)
        
        # Run on qBraid
        provider = QbraidProvider()
        qbraid_device = provider.get_device(device.value)
        
        try:
            transpiled_circuit = qbraid_transpile(qc, qbraid_device)
            job = qbraid_device.run(transpiled_circuit, shots=1024)
            result = job.result()
            counts = result.data.get_counts()
            return counts, qc
        except Exception as e:
            print(f"Error running on qBraid: {str(e)}")
            counts = AerSimulator().run(qc).result().get_counts()
            return counts, qc
    
    def run_ibm(self, channel='ibm_quantum', pre_gate='s', post_gate='sdg', token=''):
        """Run the circuit on IBM's quantum computer or simulator"""
        # qc = self.create_circuit(pre_gate, post_gate)
        qc = self.create_single_qubit_gate_fidelity(gates=['u', 's', 't', 'u', 's'])
        
        try:
            # Initialize the IBM Quantum service
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
            isa_circuit = pm.run(qc)
            job = sampler.run([isa_circuit])
            print(f">>> Job ID: {job.job_id()}")
            print(f">>> Job Status: {job.status()}")
            
            # Add timeout check
            import time
            start_time = time.time()
            while time.time() - start_time < 120:  # 2 minutes timeout
                status = job.status()
                print(f">>> Job Status: {status}")
                
                if status == 'DONE':
                    result = job.result()
                    counts = result[0].data.test_result.get_counts()
                    return counts, qc
                elif status != 'RUNNING' and status != 'QUEUED':
                    print(f"Job ended with status: {status}")
                    return 0, qc
                
                time.sleep(5)  # Check every 5 seconds
            
            # If we reach here, we've timed out (job still QUEUED or RUNNING)
            print("Job timed out after 2 minutes")
            return 0, qc
                
        except Exception as e:
            print(f"Error running on IBM: {str(e)}")
            # Fallback to simulator if there's an error
            counts = AerSimulator().run(qc).result().get_counts()
            return counts, qc
    
    def create_single_qubit_gate_fidelity(self, gates=['s'], post_gates=None):
        """Quick validation of gates by applying them and their conjugates to a single qubit
        Args:
            gates (list): List of gates to apply
            post_gates (list, optional): List of gates to apply after. If None, conjugates will be applied
        """
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'test_result')
        qc = QuantumCircuit(qr, cr)
        
        # Apply gates
        for gate in gates:
            if gate == 'u':
                qc.u(0.5, 0.3, 0.2, qr[0])
            else:
                getattr(qc, gate)(qr[0])
        
        # Apply post_gates if provided, otherwise apply conjugates
        if post_gates:
            for gate in post_gates:
                if gate == 'u':
                    qc.u(0.5, 0.3, 0.2, qr[0])  # Using same angles as before
                else:
                    getattr(qc, gate)(qr[0])
        else:
            # Apply conjugates in reverse order
            for gate in reversed(gates):
                if gate == 'u':
                    qc.u(-0.5, -0.2, -0.3, qr[0])  # Negative parameters in conjugate order
                elif gate == 's':
                    qc.sdg(qr[0])
                elif gate == 't':
                    qc.tdg(qr[0])
                else:
                    # x and y are self-inverse
                    getattr(qc, gate)(qr[0])
        
        qc.measure(qr, cr)

        return qc
