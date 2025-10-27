#!/bin/bash

echo "🎤 Voice Search Audio Snippets - Setup"
echo "======================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the backend:"
echo "  source venv/bin/activate"
echo "  uvicorn backend:app --reload"
echo ""
echo "Then open index.html in your browser."

