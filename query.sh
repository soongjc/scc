#!/bin/bash

BASE_URL="http://127.0.0.1:8000"

PDF_FILE="FAA-N26 Notice on Competency Requirements for Representatives of Financial Advisers-1.pdf"

echo "Uploading PDF to ${BASE_URL}/upload_pdf ..."
curl -X POST -F "file=@${PDF_FILE}" "${BASE_URL}/upload_pdf"
echo -e "\n\nUpload complete."

sleep 2

# QUERY="How to calculate hdb ratio?"
# QUERY_URL=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$QUERY'''))")

# echo -e "\nQuerying the indexed documents..."
# curl "${BASE_URL}/generate?q=${QUERY_URL}"
# echo -e "\n\nQuery complete."