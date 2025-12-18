#!/usr/bin/env python3
"""Health check for Backend KibaLock"""
import sys
import requests

try:
    response = requests.get("http://localhost:8505/_stcore/health", timeout=5)
    if response.status_code == 200:
        sys.exit(0)
    else:
        sys.exit(1)
except:
    sys.exit(1)
