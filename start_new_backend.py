"""
Start script for the new streamlined Logikos backend.
"""

import uvicorn
from backend.api.main import app

if __name__ == "__main__":
    print("🚀 Starting Logikos Mathematical Chat Assistant")
    print("=" * 50)
    print("New streamlined architecture v2.0")
    print("Backend API: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)