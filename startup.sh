#!/bin/bash

echo "🔧 [startup.sh] Installing requirements..."
/home/site/venv/bin/pip install --upgrade pip
/home/site/venv/bin/pip install -r /home/site/wwwroot/requirements.txt

echo "🚀 [startup.sh] Starting FastAPI app..."
/home/site/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
