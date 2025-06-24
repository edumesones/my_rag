@echo off
REM DSPy x Gradio RAG System Startup Script for Windows

echo 🚀 Starting DSPy x Gradio RAG System...

REM Check if we're in the right directory
if not exist "backend\pyproject.toml" (
    echo ❌ Error: Please run this script from the project root directory
    echo    Expected: backend\pyproject.toml
    pause
    exit /b 1
)

REM Check if OPENAI_API_KEY is set
if "%OPENAI_API_KEY%"=="" (
    echo ⚠️  Warning: OPENAI_API_KEY environment variable is not set
    echo    Please set it before starting the application:
    echo    set OPENAI_API_KEY=your-api-key-here
    echo.
    set /p continue="Do you want to continue anyway? (y/N): "
    if /i not "%continue%"=="y" (
        pause
        exit /b 1
    )
)

REM Navigate to backend directory
cd backend

REM Check if poetry is installed
poetry --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Poetry is not installed
    echo    Please install Poetry first: https://python-poetry.org/docs/#installation
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Checking dependencies...
poetry install

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir data
if not exist "chroma_db" mkdir chroma_db

REM Set default environment variables if not set
if "%DATA_DIR%"=="" set DATA_DIR=data
if "%CHROMA_DB_PATH%"=="" set CHROMA_DB_PATH=chroma_db

echo 🔧 Environment configuration:
echo    DATA_DIR: %DATA_DIR%
echo    CHROMA_DB_PATH: %CHROMA_DB_PATH%
echo    OPENAI_API_KEY: %OPENAI_API_KEY:~0,10%...

echo.
echo 🌐 Starting the application...
echo    Web UI will be available at: http://localhost:8000/gradio
echo    API docs will be available at: http://localhost:8000/docs
echo    Press Ctrl+C to stop the application
echo.

poetry run python main.py 