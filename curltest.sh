#curltest.sh
#./curltest.sh >> curltest.txt 2>&1
curl -X POST http://127.0.0.1:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "prompt": "The best thing about AI is",
  "max_tokens": 150,
  "stream": true
}'