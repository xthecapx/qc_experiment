# Quantum Code Quality Framework

A framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs). This framework helps developers and researchers understand how their quantum algorithms perform on real hardware, providing insights into QPU behavior and code quality metrics.

## Project Structure

This repository contains two main components:

1. **Main Framework** - The root directory contains:
   - A FastAPI web application in the `/app` directory that provides endpoints for running quantum experiments
   - Monitoring infrastructure with Prometheus and Grafana
   - Docker configuration for all components
   
2. **Experiment Analysis** - The `/experiment` directory contains specific analysis projects that use the framework:
   - **Circuit Depth Analysis** - Analysis of how circuit depth affects quantum algorithm performance
   - **Many Gates Results** - Analysis of quantum circuit behavior with varying numbers of gates

For specific experiment details, please see the README.md file in the `/experiment` directory.

## Overview

The framework provides:
- A web API for executing quantum circuits on QPUs
- Prometheus metrics collection for monitoring quantum experiments
- Grafana dashboards for visualizing results
- Comprehensive analysis capabilities in the `/experiment` directory
- Docker-based deployment for all components

## Core Components

### 1. Web Application (FastAPI)

The `/app` directory contains a FastAPI application with endpoints for:
- Running teleportation experiments
- Executing quantum simulations
- Testing gate fidelity
- Integrating with IBM Quantum and QBraid services

### 2. Monitoring Infrastructure

- **Prometheus**: Collects metrics about quantum experiments
- **Grafana**: Visualizes experiment results and system performance
- Custom metrics for tracking quantum circuit success rates and properties

### 3. Experiment Analysis

The `/experiment` directory contains in-depth analysis of:
- Circuit depth impact on performance
- Effects of gate count on quantum algorithm behavior
- Regression models for predicting quantum circuit performance

## Running the Framework

1. Start the entire infrastructure with Docker Compose:
```bash
docker-compose up -d
```

2. Access the FastAPI application at http://localhost:8002
3. Access Grafana dashboards at http://localhost:3000
4. Access Prometheus metrics at http://localhost:9090

## Contributing

We welcome contributions! To add your own validator:

1. Create a new validator class extending BaseValidator
2. Implement required methods
3. Add tests
4. Update documentation
5. Submit a pull request

## How to Cite

If you use this code or find it helpful in your research, please cite it using:

```bibtex 
@software{https://github.com/xthecapx/qc_experiment,
    doi = {10.5281/zenodo.14919895},
    url = {https://doi.org/10.5281/zenodo.14919895},
    year = {2024},
    publisher = {Zenodo},
    title = {Your Repository Title}
}
```

You can also simply cite the [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14919895.svg)](https://doi.org/10.5281/zenodo.14919895)
