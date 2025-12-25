#!/bin/bash
# Railway startup script
# Seeds models to persistent volume if not already present

set -e

echo "=== NFL Bets API Startup ==="

# Check if models volume is empty (first deploy)
if [ ! -f /app/models/trained/spread_model_latest.joblib ]; then
    echo "Seeding models volume from bundled models..."
    cp /app/models/bundled/*.joblib /app/models/trained/ 2>/dev/null || true
    echo "Models seeded successfully"
else
    echo "Models already present in volume"
fi

# List available models
echo "Available models:"
ls -la /app/models/trained/*.joblib 2>/dev/null || echo "  No models found"

# Start the API server
echo "Starting uvicorn..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
