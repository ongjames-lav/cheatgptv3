#!/usr/bin/env python3
"""
Check timestamp formats and what current time should be
"""

import time
from datetime import datetime

print("Current time analysis:")
print(f"Current Unix timestamp: {time.time()}")
print(f"Current date: {datetime.now()}")
print()

# Check what your session timestamps represent
session_timestamps = [1757832160.5067573, 1757830673.809938, 1757829165.446648]

print("Session timestamps analysis:")
for ts in session_timestamps:
    print(f"Timestamp: {ts}")
    print(f"  As seconds: {datetime.fromtimestamp(ts)}")
    print(f"  As milliseconds: {datetime.fromtimestamp(ts/1000)}")
    print()

print("Expected current timestamp range:")
current_real = time.time()
print(f"Real current time: {current_real} ({datetime.fromtimestamp(current_real)})")
print(f"30 days ago: {current_real - (30 * 24 * 60 * 60)} ({datetime.fromtimestamp(current_real - (30 * 24 * 60 * 60))})")