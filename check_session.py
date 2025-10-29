import sys, os
sys.path.insert(0, 'web_app')
from db_manager import DatabaseManager
from pathlib import Path

db = DatabaseManager()
session_id = 'single_1761065641'

print("=" * 60)
print(f"CHECKING SESSION: {session_id}")
print("=" * 60)

# Get session from database
s = db.get_session(session_id)
print('\nSESSION DATA FROM DATABASE:')
if s:
    for key, value in s.items():
        print(f'  {key}: {value}')
else:
    print('  ❌ Session not found in database!')

# List all matching files
print('\nFILES MATCHING SESSION ID:')
base = Path('.')
found_files = []
for p in base.rglob(f'*{session_id}*'):
    if p.is_file():
        found_files.append(p)
        print(f'  ✓ {p}')

if not found_files:
    print('  ❌ No files found matching session ID')

# Check if video_path exists
if s and s.get('video_path'):
    video_path = Path(s['video_path'])
    print(f'\nCHECKING STORED VIDEO PATH:')
    print(f'  Path: {video_path}')
    print(f'  Exists: {video_path.exists()}')
    print(f'  Absolute: {video_path.absolute()}')
    
print("\n" + "=" * 60)
