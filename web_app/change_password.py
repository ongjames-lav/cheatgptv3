"""
Change the deletion password for CheatGPT Web App
"""
import json
import os

def change_deletion_password():
    """Change the deletion password in config.json"""
    config_path = 'config.json'
    
    # Load current config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Current deletion password: {config.get('deletion_password', 'Not set')}")
    else:
        print("Config file not found. Creating new one...")
        config = {
            "deletion_password": "cheatgpt2024",
            "session_timeout_minutes": 60,
            "max_upload_size_mb": 500,
            "enable_auto_cleanup": False,
            "cleanup_days_threshold": 30
        }
    
    # Get new password
    print("\nEnter new deletion password (or press Enter to keep current):")
    new_password = input("New password: ").strip()
    
    if not new_password:
        print("❌ No password entered. Keeping current password.")
        return
    
    # Confirm password
    confirm_password = input("Confirm password: ").strip()
    
    if new_password != confirm_password:
        print("❌ Passwords don't match. Password not changed.")
        return
    
    # Update config
    config['deletion_password'] = new_password
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n✅ Deletion password updated successfully!")
    print(f"New password: {new_password}")
    print(f"\nNote: Restart the web app (app.py) for changes to take effect.")

if __name__ == '__main__':
    print("="*60)
    print("CheatGPT - Change Deletion Password")
    print("="*60)
    change_deletion_password()
