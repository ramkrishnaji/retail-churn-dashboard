#!/bin/bash

# Start FastAPI backend in the background on port 8000
echo "Starting FastAPI backend..."
uvicorn 4_api:app --host 127.0.0.1 --port 8000 &

# Start Streamlit dashboard in the foreground on port 7860
echo "Starting Streamlit dashboard..."
streamlit run 5_dashboard.py --server.port 7860 --server.address 0.0.0.0
