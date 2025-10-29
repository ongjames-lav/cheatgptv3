"""
Convert favicon.png to favicon.ico for desktop shortcut
Run this once to create the .ico file
"""
from PIL import Image
import os

# Paths
png_path = r"D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3\web_app\static\favicon.png"
ico_path = r"D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3\web_app\static\favicon.ico"

try:
    # Open the PNG file
    img = Image.open(png_path)
    
    # Convert to ICO (multiple sizes for better quality)
    img.save(ico_path, format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    
    print("✅ SUCCESS: favicon.ico created!")
    print(f"Location: {ico_path}")
    print("\nNow run 'Create_Desktop_Shortcut.vbs' to create your desktop shortcut.")
    
except FileNotFoundError:
    print(f"❌ ERROR: favicon.png not found at {png_path}")
    print("Please check the path and try again.")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: just copy and rename (Windows will handle it)
    try:
        img = Image.open(png_path)
        img.save(ico_path, format='ICO')
        print("✅ SUCCESS: favicon.ico created using alternative method!")
    except Exception as e2:
        print(f"❌ Failed: {e2}")

input("\nPress Enter to close...")
