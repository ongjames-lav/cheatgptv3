"""Verify Supervision ByteTrack is actively working in detection pipeline."""
import cv2
import logging
from cheatgpt.engines.engine_hybrid import EngineHybrid

# Setup logging to see all ByteTrack activity
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def main():
    print("=" * 80)
    print("BYTETRACK VERIFICATION TEST")
    print("=" * 80)
    print("\nThis test will verify that Supervision's ByteTrack is actively tracking")
    print("persons across frames with persistent IDs.\n")
    
    # Initialize engine
    print("🚀 Initializing EngineHybrid...")
    engine = EngineHybrid()
    print("✅ Engine initialized\n")
    
    # Open webcam
    print("📹 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return
    
    print("✅ Webcam opened")
    print("\n" + "=" * 80)
    print("WATCHING FOR BYTETRACK ACTIVITY...")
    print("=" * 80)
    print("\nLook for these log messages proving ByteTrack is working:")
    print("  🔄 BYTETRACK INPUT: X person detections")
    print("  ✅ BYTETRACK OUTPUT: X persons tracked | IDs: [1, 2, 3...]")
    print("  ✅ BYTETRACK VERIFIED: All persons have persistent track IDs")
    print("  🎯 POSE ANALYSIS INPUT: X tracked persons with ByteTrack IDs: [...]")
    print("  ✅ POSE ANALYSIS OUTPUT: X poses with ByteTrack IDs: [...]")
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame through engine (ByteTrack runs here)
            overlay_frame, events = engine.process_frame(frame)
            
            # Display frame
            cv2.imshow('ByteTrack Verification', overlay_frame)
            
            # Show instructions on first frame
            if frame_count == 1:
                print("\n📺 Video window opened. Check console logs above for ByteTrack activity.\n")
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 80)
        print("VERIFICATION COMPLETE")
        print("=" * 80)
        print("\nIf you saw the log messages listed above, ByteTrack is working correctly!")
        print("The track IDs should remain consistent for the same person across frames.\n")

if __name__ == "__main__":
    main()
