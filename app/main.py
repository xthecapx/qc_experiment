from typing import List
from typing_extensions import Annotated
from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
from app.quantum.simulator import setup_simulator
from functools import lru_cache
from app.config import Settings
import ast
from app.quantum.teleportation import teleportation_experiment, qbraid_teleportation_experiment

app = FastAPI()

@lru_cache
def get_settings():
    return Settings()

class Probabilities(BaseModel):
    probabilities: List[int]


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
def execute_teleportation(executions: int = Query(..., description="Number of executions for the teleportation experiment")):
    if executions <= 0:
        raise ValueError("Number of executions must be a positive integer")

    success_rate, counts = teleportation_experiment(executions)
    
    return {
        "success_rate": success_rate,
        "counts": counts
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