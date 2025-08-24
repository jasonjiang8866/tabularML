@echo off
REM Auto setup script using UV package manager for Windows
REM This script sets up the environment and installs dependencies for TabularML

echo 🚀 Setting up TabularML environment with UV...

REM Check if UV is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing UV package manager...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
)

echo ✅ UV installed successfully

REM Initialize UV project if not already initialized
if not exist "uv.lock" (
    echo 🔧 Initializing UV project...
    uv sync
)

echo 📚 Installing dependencies...
uv sync

echo 🎉 Environment setup complete!
echo.
echo Available commands:
echo   uv run streamlit run ui.py          # Run the Streamlit UI
echo   uv run python pipeline.py          # Run the ML pipeline directly
echo   uv run python -m pytest            # Run tests (if available)
echo.
echo To activate the environment manually:
echo   .venv\Scripts\activate