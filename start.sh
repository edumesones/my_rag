#!/bin/bash

# DSPy x Gradio RAG System Startup Script

echo "🚀 Starting DSPy x Gradio RAG System..."

# Check if we're in the right directory
if [ ! -f "backend/pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected: backend/pyproject.toml"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set"
    echo "   Please set it before starting the application:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Navigate to backend directory
cd backend

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry is not installed"
    echo "   Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
poetry install

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p chroma_db

# Set default environment variables if not set
export DATA_DIR=${DATA_DIR:-"data"}
export CHROMA_DB_PATH=${CHROMA_DB_PATH:-"chroma_db"}

echo "🔧 Environment configuration:"
echo "   DATA_DIR: $DATA_DIR"
echo "   CHROMA_DB_PATH: $CHROMA_DB_PATH"
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."

# Start the application
echo ""
echo "🌐 Starting the application..."
echo "   Web UI will be available at: http://localhost:8000/gradio"
echo "   API docs will be available at: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop the application"
echo ""

poetry run python main.py 