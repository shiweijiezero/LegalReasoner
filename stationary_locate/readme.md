# Deployment
```bash
pip install fastapi uvicorn
python trace_clause_gpt_fastapi.py
uvicorn trace_clause_gpt_fastapi:app --host 0.0.0.0 --port 8123
```
# Inference
Bash
```bash
curl -X 'POST' \
  'http://dgx-021:8123/analyze' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "text": "How can I create a will or trust?"
}'
```
Python
```python
import requests

response = requests.post(
    "http://dgx-021:8123/analyze",
    json={"text": "How can I create a will or trust?"}
)
print(response.json())
```