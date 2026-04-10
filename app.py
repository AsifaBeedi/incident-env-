# app.py - Root level entry point
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server.app import app

def main():
    """Entry point for Hugging Face Spaces."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
