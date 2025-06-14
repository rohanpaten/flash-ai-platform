#!/bin/bash

# FLASH Local Development Startup Script
# Run this to start FLASH locally for testing

echo "🚀 Starting FLASH Local Development Environment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to kill process on port
kill_port() {
    echo -e "${YELLOW}Killing existing process on port $1...${NC}"
    lsof -ti :$1 | xargs kill -9 2>/dev/null || true
    sleep 2
}

echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"

# Check Python
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    echo "Install from: https://www.python.org/downloads/"
    exit 1
fi

# Check Node.js
if ! command_exists node; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 16+${NC}"
    echo "Install from: https://nodejs.org/"
    exit 1
fi

# Check npm
if ! command_exists npm; then
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites met${NC}"

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}Step 2: Setting up backend...${NC}"

# Install lightweight Python dependencies
if [ ! -f "requirements_lightweight.txt" ]; then
    echo -e "${RED}❌ requirements_lightweight.txt not found${NC}"
    exit 1
fi

echo "Installing Python dependencies..."
python3 -m pip install -r requirements_lightweight.txt

# Check if lightweight API server exists
if [ ! -f "api_server_lightweight.py" ]; then
    echo -e "${RED}❌ api_server_lightweight.py not found${NC}"
    exit 1
fi

echo -e "${BLUE}Step 3: Setting up frontend...${NC}"

# Navigate to frontend directory
if [ ! -d "flash-frontend-apple" ]; then
    echo -e "${RED}❌ flash-frontend-apple directory not found${NC}"
    exit 1
fi

cd flash-frontend-apple

# Install Node dependencies
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json not found in frontend directory${NC}"
    exit 1
fi

echo "Installing Node.js dependencies..."
npm install

# Create local environment file
echo "REACT_APP_API_URL=http://localhost:8001" > .env.local

cd ..

echo -e "${BLUE}Step 4: Starting services...${NC}"

# Kill existing processes on required ports
if port_in_use 8001; then
    kill_port 8001
fi

if port_in_use 3000; then
    kill_port 3000
fi

# Start backend in background
echo -e "${YELLOW}Starting backend API server on port 8001...${NC}"
python3 api_server_lightweight.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! port_in_use 8001; then
    echo -e "${RED}❌ Backend failed to start. Check backend.log${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✅ Backend started successfully${NC}"

# Test backend health
if command_exists curl; then
    echo "Testing backend health..."
    if curl -s http://localhost:8001/health > /dev/null; then
        echo -e "${GREEN}✅ Backend health check passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Backend health check failed, but continuing...${NC}"
    fi
fi

# Start frontend
echo -e "${YELLOW}Starting frontend on port 3000...${NC}"
cd flash-frontend-apple

# Start frontend in background
npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!

cd ..

# Wait for frontend to start
echo "Waiting for frontend to start..."
sleep 10

# Check if frontend is running
if ! port_in_use 3000; then
    echo -e "${RED}❌ Frontend failed to start. Check frontend.log${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✅ Frontend started successfully${NC}"

# Save PIDs for cleanup
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

echo ""
echo -e "${GREEN}🎉 FLASH is now running locally!${NC}"
echo "=============================================="
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}Backend:${NC}  http://localhost:8001"
echo -e "${BLUE}Health:${NC}   http://localhost:8001/health"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo -e "  Backend: ${SCRIPT_DIR}/backend.log"
echo -e "  Frontend: ${SCRIPT_DIR}/frontend.log"
echo ""
echo -e "${YELLOW}To stop:${NC} ./stop_local.sh"
echo ""
echo -e "${GREEN}Your browser should open automatically...${NC}"

# Try to open browser
sleep 3
if command_exists open; then
    open http://localhost:3000
elif command_exists xdg-open; then
    xdg-open http://localhost:3000
elif command_exists start; then
    start http://localhost:3000
fi

echo -e "${BLUE}Press Ctrl+C to view logs or use ./stop_local.sh to stop services${NC}"

# Wait for user input or signals
trap 'echo -e "\n${YELLOW}Use ./stop_local.sh to stop services${NC}"; exit 0' INT TERM

# Keep script running
tail -f /dev/null