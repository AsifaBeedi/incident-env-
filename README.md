# Incident Response Environment for AI Agents

This project implements an OpenEnv-compliant simulation environment for training and evaluating AI agents in incident response scenarios. The environment models a distributed microservices architecture where agents must diagnose system faults and apply appropriate remediation actions.

## Project Structure

```
project-root/
├── env/
│   ├── models.py          # Data models and type definitions
│   ├── env.py             # Core environment implementation
│   ├── tasks.py           # Task definitions and reward logic
│   └── __init__.py        # Package initialization
├── inference.py           # LLM agent inference loop
├── demo.py                # Demonstration and testing script
├── openenv.yaml           # Environment specification
├── Dockerfile             # Containerization configuration
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## Core Components

### Environment (`env/`)

- **models.py**: Defines action types, observation structures, and service state representations
- **env.py**: Implements the IncidentResponseEnv class with reset/step/state methods
- **tasks.py**: Contains task definitions, fault injection logic, and reward calculation

### Inference Engine

- **inference.py**: Handles LLM API integration, structured output formatting, and episode execution
- Supports OpenAI-compatible APIs with configurable endpoints and models

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AsifaBeedi/incident-env-.git
   cd incident-env-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Local Testing

Run the demonstration script to verify environment functionality:
```bash
python demo.py
```

### Inference Execution

Set required environment variables:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token"
```

Execute the inference loop:
```bash
python inference.py
```

### Docker Deployment

Build and run the containerized environment:
```bash
docker build -t incident-env .
docker run -e API_BASE_URL="..." -e MODEL_NAME="..." -e HF_TOKEN="..." incident-env
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint URL | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier for inference | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | Authentication token for API access | Required |

## Tasks

The environment includes three difficulty levels:

1. **auth-crash-loop**: Basic service restart scenario
2. **payments-oom-cascade**: Multi-service failure with cascading effects
3. **network-split-db-leak**: Complex distributed system failure

## Output Format

The inference script produces structured stdout output following OpenEnv specifications:

```
[START] task=task_name env=IncidentResponseEnv model=model_name
[STEP] step=1 action=action_name reward=0.25 done=false error=null
[END] success=true steps=3 rewards=0.25,0.35,0.55
```

## Technical Specifications

- **Python Version**: 3.11+
- **Dependencies**: pydantic, openai
- **Reward Range**: [0.0, 1.0] per step
- **Runtime Constraints**: Maximum 20 minutes execution time
- **Resource Limits**: 2 vCPU, 8GB memory

## Development

### Code Quality

- Type hints throughout codebase
- Comprehensive error handling
- Structured logging and output formatting
- Modular architecture with clear separation of concerns

### Testing

Run the demo script to validate environment behavior:
```bash
python demo.py
```

### Containerization

The project includes Docker support for reproducible deployment across environments.

## License

This project is part of the OpenV Hack competition and follows the specified submission requirements.