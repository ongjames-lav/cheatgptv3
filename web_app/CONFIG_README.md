# CheatGPT Configuration Guide

## Changing the Deletion Password

The deletion password is used to confirm when deleting sessions from the analytics page. You can change this password in two ways:

### Method 1: Using the Password Change Script (Recommended)

1. Open terminal in the `web_app` folder
2. Run the password change script:
   ```bash
   python change_password.py
   ```
3. Enter your new password when prompted
4. Confirm the password
5. Restart the web app for changes to take effect

### Method 2: Manually Edit config.json

1. Open `web_app/config.json` in a text editor
2. Change the `deletion_password` value:
   ```json
   {
       "deletion_password": "your_new_password_here",
       ...
   }
   ```
3. Save the file
4. Restart the web app

## Configuration Options

The `config.json` file contains the following settings:

- **deletion_password**: Password required to delete sessions (default: `cheatgpt2024`)
- **session_timeout_minutes**: Session timeout in minutes (default: `60`)
- **max_upload_size_mb**: Maximum upload file size in MB (default: `500`)
- **enable_auto_cleanup**: Automatically cleanup old sessions (default: `false`)
- **cleanup_days_threshold**: Days before sessions are auto-cleaned (default: `30`)

## Security Notes

- The deletion password is stored in plain text in `config.json`
- Keep this file secure and don't commit it to public repositories
- Choose a strong password for production use
- The web app must be restarted after changing configuration

## Default Password

The default deletion password is: **cheatgpt2024**

We recommend changing this immediately after installation.
