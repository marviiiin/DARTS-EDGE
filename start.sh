#!/bin/bash
# Set your API keys here before running
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
cd /home/moseikuffour/darts_edge
source venv/bin/activate
exec python app.py
