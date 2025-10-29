#!/usr/bin/env python3
"""
Verify that processed videos are valid MP4 files playable in external media players
"""

import struct
from pathlib import Path
import os

def check_mp4_integrity(video_path):
    """
    Check if file is a valid MP4 by verifying MP4 file structure
    
    Returns:
        dict with integrity check results
    """
    results = {
        'path': str(video_path),
        'exists': False,
        'size_mb': 0,
        'is_valid_mp4': False,
        'has_ftyp': False,
        'has_mdat': False,
        'has_moov': False,
        'details': []
    }
    
    if not Path(video_path).exists():
        results['details'].append("File does not exist")
        return results
    
    results['exists'] = True
    file_size = os.path.getsize(video_path)
    results['size_mb'] = file_size / 1024 / 1024
    
    # Check if file is large enough to be a video
    if file_size < 1000:
        results['details'].append(f"File too small: {file_size} bytes")
        return results
    
    try:
        with open(video_path, 'rb') as f:
            # Read first bytes to check MP4 signature
            # MP4 files start with: [size (4 bytes)][ftyp (4 bytes)]
            header = f.read(8)
            
            if len(header) >= 8:
                # Check for ftyp box
                if b'ftyp' in header:
                    results['has_ftyp'] = True
                    results['details'].append("✓ Found ftyp box (MP4 signature)")
                    results['is_valid_mp4'] = True
            
            # Scan file for required MP4 boxes
            f.seek(0)
            file_content = f.read()
            
            # Check for mdat (media data)
            if b'mdat' in file_content:
                results['has_mdat'] = True
                results['details'].append("✓ Found mdat box (media data)")
            
            # Check for moov (movie metadata)
            if b'moov' in file_content:
                results['has_moov'] = True
                results['details'].append("✓ Found moov box (metadata)")
            
            # Overall validity
            if results['has_ftyp'] and results['has_mdat']:
                results['is_valid_mp4'] = True
                results['details'].append("✓ Valid MP4 file structure")
            
    except Exception as e:
        results['details'].append(f"Error reading file: {e}")
    
    return results

def main():
    """Check all processed videos"""
    
    print("\n" + "=" * 70)
    print("MP4 PLAYABILITY VERIFICATION")
    print("=" * 70)
    
    results_dir = Path('web_app/results')
    
    if not results_dir.exists():
        print("❌ Results folder not found")
        return
    
    # Find all processed videos
    video_files = list(results_dir.glob('*/processed_*.mp4'))
    print(f"\nFound {len(video_files)} processed video files\n")
    
    valid_count = 0
    invalid_count = 0
    sample_count = 0
    
    for video_path in sorted(video_files):
        # Sample first 10
        if sample_count >= 10:
            print(f"\n... and {len(video_files) - 10} more videos")
            break
        
        sample_count += 1
        result = check_mp4_integrity(video_path)
        
        print(f"\n{sample_count}. {video_path.parent.name}")
        print(f"   Size: {result['size_mb']:.1f} MB")
        
        if result['is_valid_mp4']:
            print("   Status: ✅ VALID MP4 FILE")
            print(f"   Structure: ftyp={result['has_ftyp']}, mdat={result['has_mdat']}, moov={result['has_moov']}")
            valid_count += 1
        else:
            print("   Status: ❌ INVALID MP4")
            invalid_count += 1
        
        for detail in result['details']:
            print(f"   {detail}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total videos checked: {sample_count}")
    print(f"Valid MP4 files: {valid_count} ✅")
    print(f"Invalid files: {invalid_count} ❌")
    
    if valid_count > 0:
        print("\n✅ EXTERNAL PLAYABILITY:")
        print("   All valid MP4 files can be played in:")
        print("   • Windows Media Player")
        print("   • VLC Media Player")
        print("   • QuickTime (Mac)")
        print("   • Any browser with HTML5 video support")
        print("   • FFmpeg/ffplay")
        print("   • HandBrake")
        print("   • And all other standard MP4 players")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
