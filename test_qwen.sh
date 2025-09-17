#!/bin/bash

echo "🧪 Testing Logikos with Qwen AI..."

# Test health endpoint
echo "1. Testing health endpoint..."
health_response=$(curl -s http://localhost:8000/health)
echo "Health: $health_response"

# Test simple math problem
echo "2. Testing simple derivative..."
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"derivative of x^2"}' \
  | jq '.reply_text, .solution.final_answer'

echo "3. API is ready at: http://localhost:8000"
echo "4. API docs available at: http://localhost:8000/docs"