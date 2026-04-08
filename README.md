# AI Incident Response OpenEnv

This project simulates an incident response environment for AI agents to practice diagnosing and fixing service issues.

## Structure

- `env/`: Core environment modules
  - `models.py`: Data models for actions and observations
  - `env.py`: The main incident response environment
  - `tasks.py`: Task definitions and configurations
- `inference.py`: LLM agent inference loop
- `demo.py`: Demonstration script
- `openenv.yaml`: Environment configuration
- `requirements.txt`: Python dependencies

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Run demo: `python demo.py`
3. Run inference: `python inference.py`

## Environment Variables

Set the following for inference:
- `API_BASE_URL`: API endpoint (default: HuggingFace)
- `MODEL_NAME`: Model name (default: Qwen/Qwen2.5-72B-Instruct)
- `HF_TOKEN`: HuggingFace token