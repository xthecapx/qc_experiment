from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qc_experiment.app.quantum.circuit import quantum_algorithm
from qiskit_ibm_runtime import SamplerV2 as Sampler

def check_ibm_status(job):
    id = job.job_id()
    status = job.status()
    print(f">>> Job ID: {id}")
    print(f">>> Job Status: {status}")

    return { "id": id, "status": status }

def setup_ibm(probabilities):
    QiskitRuntimeService.save_account(
        channel='ibm_quantum',
        token = '',
        overwrite=True
    )
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    print(backend.configuration().backend_name)
    print(backend.status().pending_jobs)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    circuit = quantum_algorithm(probabilities) # 1x1x
    isa_circuit = pm.run(circuit)
    sampler = Sampler(backend)
    
    job = sampler.run([isa_circuit])

    return job

        
    # isa_circuit.draw('mpl', idle_wires=False)