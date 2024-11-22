import os
from typing import List
from typing_extensions import Annotated
from fastapi import FastAPI, Depends, Query, Path
from pydantic import BaseModel
from app.quantum.simulator import setup_simulator
from functools import lru_cache
from app.config import Settings
import ast
from app.quantum.teleportation import teleportation_experiment, qbraid_teleportation_experiment
from prometheus_fastapi_instrumentator import Instrumentator
from app.utils import PrometheusMiddleware, metrics, setting_otlp
from prometheus_client import Gauge
from app.quantum.teleportation_antiteleportation import TeleportationValidator
from enum import Enum
from app.quantum.teleportation_gate_fidelity import GateFidelity, QbraidDevice
from qiskit_ibm_runtime import QiskitRuntimeService

teleportation_success_rate = Gauge('teleportation_success_rate', 'Success rate of teleportation experiment', ['executions', 'num_gates', 'depth'])
teleportation_executions = Gauge('teleportation_executions', 'Number of executions for teleportation experiment')
teleportation_num_gates = Gauge('teleportation_num_gates', 'Number of gates in teleportation circuit')
teleportation_circuit_depth = Gauge('teleportation_circuit_depth', 'Depth of the teleportation circuit')

app = FastAPI()

APP_NAME = os.environ.get("APP_NAME", "app")

app.add_middleware(PrometheusMiddleware, app_name=APP_NAME)
Instrumentator().instrument(app).expose(app)

@lru_cache
def get_settings():
    return Settings()

class Probabilities(BaseModel):
    probabilities: List[int]


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/probabilities/")
def read_probabilities(body: Probabilities, settings: Annotated[Settings, Depends(get_settings)]):
    outcome, qc, counts = setup_simulator(body.probabilities)
    circuit_str = qc.draw(output='text').single_string()
    circuit_rows = circuit_str.split('\n')
    print(settings.token)

    return {"outcome": outcome, "counts": counts, "circuit": circuit_rows  }

@app.get("/settings/")
def get_current_settings(settings: Annotated[Settings, Depends(get_settings)]):
    return settings

class SimulatorInput(BaseModel):
    probabilities: List[float]

@app.get("/simulator/")
def execute_simulator(probabilities: str = Query(..., description="List of probabilities for each slot (0-100), e.g. [50,50,50]")):
    try:
        prob_list = ast.literal_eval(probabilities)
        if not isinstance(prob_list, list) or not all(isinstance(x, (int, float)) for x in prob_list):
            raise ValueError("Invalid input format")
    except:
        raise ValueError("Invalid input format. Please provide a list of numbers, e.g. [50,50,50]")

    outcome, qc, counts = setup_simulator(prob_list)
    circuit_str = qc.draw(output='text').single_string()
    
    return {
        "outcome": outcome,
        "circuit": circuit_str,
        "counts": counts
    }

@app.get("/teleportation/")
def execute_teleportation(
    num_gates: int = Query(1, description="Number of gates in the teleportation circuit"),
    num_payload_qubits: int = Query(1, description="Number of payload qubits to teleport")
):
    if num_gates < 1:
        raise ValueError("Number of gates must be at least 1")
    if num_payload_qubits < 1:
        raise ValueError("Number of payload qubits must be at least 1")

    # Create validator with specified parameters
    validator = TeleportationValidator(payload_size=num_payload_qubits, num_gates=num_gates)
    
    # Get metrics from the simulation
    metrics = validator.get_metrics()
    
    # Update Prometheus metrics
    teleportation_success_rate.labels(
        executions=1,  # Default to 1 since we're not using executions
        num_gates=num_gates,
        depth=metrics["circuit_metrics"]["depth"]
    ).set(metrics["results_metrics"]["success_rate"])
    teleportation_num_gates.set(num_gates)
    teleportation_circuit_depth.set(metrics["circuit_metrics"]["depth"])
    
    return {
        "results_metrics": {
            "success_rate": metrics["results_metrics"]["success_rate"],
            "counts": metrics["results_metrics"]["counts"],
        },
        "circuit_metrics": {
            "depth": metrics["circuit_metrics"]["depth"],
            "width": metrics["circuit_metrics"]["width"],
            "size": metrics["circuit_metrics"]["size"],
            "count_ops": metrics["circuit_metrics"]["count_ops"],
        },
        "config_metrics": metrics["config_metrics"],
        "custom_gate_distribution": metrics["custom_gate_distribution"]
    }

@app.get("/qbraid-teleportation/")
def execute_qbraid_teleportation(
    executions: int = Query(..., description="Number of executions for the qBraid teleportation experiment"),
):
    if executions <= 0:
        raise ValueError("Number of executions must be a positive integer")

    # Pass the API key from settings to the experiment function
    success_rate, counts = qbraid_teleportation_experiment(executions)
    
    return {
        "success_rate": success_rate,
        "counts": counts
    }

class GateType(str, Enum):
    S = "s"
    SDG = "sdg"
    T = "t"
    TDG = "tdg"
    Z = "z"
    H = "h"

class SimulatorType(str, Enum):
    QBRAID = "qbraid"
    LOCAL = "local"
    IBM = "ibm"

@app.post("/gate-fidelity/")
def run_gate_fidelity(
    settings: Annotated[Settings, Depends(get_settings)],
    pre_gate: GateType = Query(GateType.S, description="Gate to apply before teleportation"),
    post_gate: GateType = Query(GateType.SDG, description="Gate to apply after teleportation"),
    device: QbraidDevice = Query(QbraidDevice.QIR, description="QBraid device to run the simulation on"),
    simulator: SimulatorType = Query(SimulatorType.LOCAL, description="Simulator to use: qbraid, local, or ibm")
):
    fidelity_test = GateFidelity()
    
    if simulator == SimulatorType.QBRAID:
        counts, circuit = fidelity_test.run_qbraid(
            device=device,
            pre_gate=pre_gate.value,
            post_gate=post_gate.value
        )
    elif simulator == SimulatorType.IBM:
        counts, circuit = fidelity_test.run_ibm(
            channel='ibm_quantum',
            pre_gate=pre_gate.value,
            post_gate=post_gate.value,
            token=settings.token
        )
    else:
        counts, circuit = fidelity_test.run_simulation(
            pre_gate=pre_gate.value,
            post_gate=post_gate.value
        )
    
    circuit_str = circuit.draw(output='text').single_string()
    
    return {
        "counts": counts,
        "circuit": circuit_str,
        "configuration": {
            "pre_gate": pre_gate.value,
            "post_gate": post_gate.value,
            "device": device.value if simulator == SimulatorType.QBRAID else "local_simulator",
            "simulator": "qbraid" if simulator == SimulatorType.QBRAID else "aer"
        }
    }

@app.get("/results/")
def get_results(
    settings: Annotated[Settings, Depends(get_settings)]
):
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token=settings.token
    )
    job = service.job('cwzt185z326g008qpbm0')
    job_result = job.result()