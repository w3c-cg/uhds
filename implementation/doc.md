# 1. Install dependencies
pip install fastapi uvicorn pydantic sqlalchemy cryptography pyjwt numpy redis

# 2. Run the server
python demo.py

# 3. Generate demo keys (optional)
python demo.py --generate-keys

# 4. Test the API
curl http://localhost:8000/health
curl http://localhost:8000/api/docs

# 5. Get a demo token
curl -X POST "http://localhost:8000/api/demo/token?user_type=patient"

# 6. Create a consent (using the token from step 5)
curl -X POST "http://localhost:8000/api/consent" \
  -H "Authorization: Bearer YOUR_TOKEN_FROM_STEP_5" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_did": "did:patient:demo_123",
    "purpose": "research",
    "data_types": ["LabResult", "Diagnosis"],
    "processing_type": "aggregation"
  }'