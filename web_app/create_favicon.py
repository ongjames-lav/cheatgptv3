#!/usr/bin/env python3
"""
Convert SVG favicon to PNG and ICO formats for better browser compatibility
"""

try:
    from PIL import Image, ImageDraw
    import cairosvg
    import io
    import os
except ImportError:
    print("Installing required packages...")
    import subprocess
    import sys
    
    packages = ['Pillow', 'cairosvg']
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
    
    # Try importing again
    try:
        from PIL import Image, ImageDraw
        import cairosvg
        import io
        import os
    except ImportError:
        print("Could not install required packages. Using fallback method.")
        cairosvg = None

def create_png_favicon():
    """Convert SVG to PNG format"""
    svg_path = "static/favicon.svg"
    png_path = "static/favicon.png"
    
    if not os.path.exists(svg_path):
        print(f"SVG file not found: {svg_path}")
        return False
    
    try:
        if cairosvg:
            # Convert SVG to PNG using cairosvg
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=32, output_height=32)
            print(f"✅ Created PNG favicon: {png_path}")
            return True
        else:
            print("❌ cairosvg not available, using fallback method")
            return create_simple_favicon()
    except Exception as e:
        print(f"❌ Error converting SVG to PNG: {e}")
        return create_simple_favicon()

def create_simple_favicon():
    """Create a simple PNG favicon as fallback"""
    try:
        # Create a simple 32x32 favicon with camera theme
        size = 32
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a camera aperture-like circle
        margin = 2
        circle_bbox = [margin, margin, size-margin, size-margin]
        
        # Outer circle (black)
        draw.ellipse(circle_bbox, fill=(0, 0, 0, 255), outline=(64, 64, 64, 255), width=2)
        
        # Inner circle (dark gray)
        inner_margin = 8
        inner_bbox = [inner_margin, inner_margin, size-inner_margin, size-inner_margin]
        draw.ellipse(inner_bbox, fill=(32, 32, 32, 255))
        
        # Red play triangle in center
        center = size // 2
        triangle_size = 6
        triangle = [
            (center - triangle_size//2, center - triangle_size//2),
            (center - triangle_size//2, center + triangle_size//2),
            (center + triangle_size//2, center)
        ]
        draw.polygon(triangle, fill=(255, 0, 0, 255))
        
        # Save PNG
        img.save("static/favicon.png", "PNG")
        print("✅ Created simple PNG favicon: static/favicon.png")
        
        # Save ICO
        img.save("static/favicon.ico", "ICO", sizes=[(32, 32)])
        print("✅ Created ICO favicon: static/favicon.ico")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating simple favicon: {e}")
        return False

if __name__ == "__main__":
    print("🎨 Creating favicon in multiple formats...")
    
    # Try to convert SVG to PNG
    if not create_png_favicon():
        print("Falling back to simple favicon creation...")
        create_simple_favicon()
    else:
        # If PNG was created successfully, also create ICO
        try:
            img = Image.open("static/favicon.png")
            img.save("static/favicon.ico", "ICO", sizes=[(32, 32)])
            print("✅ Created ICO favicon: static/favicon.ico")
        except Exception as e:
            print(f"❌ Error creating ICO: {e}")
    
    print("\n🎉 Favicon setup complete!")
    print("Your website will now show the custom favicon in browser tabs.")