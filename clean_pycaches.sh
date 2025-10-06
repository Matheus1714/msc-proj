#!/bin/bash

echo "Removendo todas as pastas __pycache__ a partir de $(pwd)..."
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "✅ Todos os __pycache__ foram removidos."
