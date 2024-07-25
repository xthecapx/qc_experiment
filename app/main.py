from typing import List
from typing_extensions import Annotated
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from app.quantum.simulator import setup_simulator
from functools import lru_cache
from app.config import Settings


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