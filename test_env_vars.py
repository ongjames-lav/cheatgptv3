"""Test environment variable loading"""

import os
from dotenv import load_dotenv

# Load from project root
load_dotenv(dotenv_path='.env')

print("🔧 Environment Variable Test")
print("=" * 30)

# Test policy variables
print("Policy Configuration:")
print(f"  BEHAVIOR_REPEAT_WINDOW: {os.getenv('BEHAVIOR_REPEAT_WINDOW', 'NOT SET')}")
print(f"  ALERT_PERSIST_FRAMES: {os.getenv('ALERT_PERSIST_FRAMES', 'NOT SET')}")
print(f"  PHONE_REPEAT_THRESH: {os.getenv('PHONE_REPEAT_THRESH', 'NOT SET')}")
print(f"  DEBUG_POLICY: {os.getenv('DEBUG_POLICY', 'NOT SET')}")

print("\nPose Configuration:")
print(f"  LEAN_ANGLE_THRESH: {os.getenv('LEAN_ANGLE_THRESH', 'NOT SET')}")
print(f"  HEAD_TURN_THRESH: {os.getenv('HEAD_TURN_THRESH', 'NOT SET')}")
print(f"  PHONE_IOU_THRESH: {os.getenv('PHONE_IOU_THRESH', 'NOT SET')}")
print(f"  DEBUG_POSE: {os.getenv('DEBUG_POSE', 'NOT SET')}")

print("\nEngine Configuration:")
print(f"  DEBUG_ENGINE: {os.getenv('DEBUG_ENGINE', 'NOT SET')}")

# Check if .env file exists
if os.path.exists('.env'):
    print("\n✅ .env file found in current directory")
    with open('.env', 'r') as f:
        lines = f.readlines()
    print(f"   Contains {len(lines)} lines")
else:
    print("\n❌ .env file not found in current directory")

print(f"\nCurrent working directory: {os.getcwd()}")
