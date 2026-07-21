import cv2
import os
from datetime import datetime

def create_capture_session(output_base_dir="datasets/custom_dataset", target_count=100):
    """
    Interactive image capture session for building custom ASL dataset.
    """
    # Create base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("=" * 60)
    print("ASL Image Capture Tool")
    print("=" * 60)
    print(f"\nImages will be saved to: {output_base_dir}/")
    print("\nControls:")
    print("  SPACE - Capture image")
    print("  Q     - Quit current class")
    print("  ESC   - Exit program")
    print("-" * 60)
    
    while True:
        # Get class name from user
        class_name = input("\nEnter class name (or 'exit' to quit): ").strip()
        
        if class_name.lower() == 'exit':
            print("\nCapture session ended!")
            break
        
        if not class_name:
            print("Please enter a valid class name")
            continue
        
        # Create class directory
        class_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Count existing images
        existing_images = len([f for f in os.listdir(class_dir) 
                              if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\nCapturing images for class: '{class_name}'")
        print(f"   Existing images: {existing_images}")
        print(f"   Target: {target_count}")
        print(f"\n   Position your hand and press SPACE to capture...")
        
        # Start webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            continue
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        count = existing_images
        
        # Define ROI (Region of Interest)
        roi_x, roi_y = 100, 50
        roi_size = 300
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), 
                         (roi_x + roi_size, roi_y + roi_size), 
                         (0, 255, 0), 3)
            
            # Add text overlays
            cv2.putText(frame, f"Class: {class_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{target_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Next class", (10, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Extract ROI for preview
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Show preview of what will be saved
            preview_size = 100
            preview = cv2.resize(roi, (preview_size, preview_size))
            frame[10:10+preview_size, frame.shape[1]-preview_size-10:frame.shape[1]-10] = preview
            cv2.putText(frame, "Preview", (frame.shape[1]-preview_size-5, 5+preview_size+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('ASL Image Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - capture
                # Generate unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{class_name}_{timestamp}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                # Save the ROI (cropped hand region)
                cv2.imwrite(filepath, roi)
                count += 1
                
                # Visual feedback - flash green
                flash_frame = frame.copy()
                cv2.rectangle(flash_frame, (roi_x, roi_y), 
                             (roi_x + roi_size, roi_y + roi_size), 
                             (0, 255, 0), -1)
                cv2.imshow('ASL Image Capture', flash_frame)
                cv2.waitKey(100)
                
                print(f"   📸 Captured: {count}/{target_count}")
                
                if count >= target_count:
                    print(f"\n   Target reached for class '{class_name}'!")
                    break
                    
            elif key == ord('q') or key == ord('Q'):  # Q - next class
                print(f"\n   Finished class '{class_name}' with {count} images")
                break
                
            elif key == 27:  # ESC - exit program
                cap.release()
                cv2.destroyAllWindows()
                print("\nCapture session ended!")
                return
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    
    if os.path.exists(output_base_dir):
        total_images = 0
        for class_name in sorted(os.listdir(output_base_dir)):
            class_path = os.path.join(output_base_dir, class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
                total_images += num_images
                print(f"   {class_name}: {num_images} images")
        
        print("-" * 60)
        print(f"   Total: {total_images} images")
        print(f"   Location: {os.path.abspath(output_base_dir)}")


def batch_capture_mode(output_base_dir="datasets/custom_dataset", classes=None, images_per_class=50):
    """
    Batch capture mode - captures images for multiple predefined classes.
    """
    if classes is None:
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 
                   'X', 'Y']  # Note: J and Z require motion
    
    print("=" * 60)
    print("ASL Batch Capture Mode")
    print("=" * 60)
    print(f"\nClasses to capture: {len(classes)}")
    print(f"Images per class: {images_per_class}")
    print(f"Total images needed: {len(classes) * images_per_class}")
    print("\nPress any key when ready to start each class...")
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    for i, class_name in enumerate(classes, 1):
        print(f"\n{'='*40}")
        print(f"Class {i}/{len(classes)}: '{class_name}'")
        print(f"{'='*40}")
        
        input(f"Position your hand to show '{class_name}' and press ENTER...")
        
        class_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        roi_x, roi_y, roi_size = 100, 50, 300
        
        print("Capturing... (Press Q to skip)")
        
        while count < images_per_class:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Draw ROI
            cv2.rectangle(frame, (roi_x, roi_y), 
                         (roi_x + roi_size, roi_y + roi_size), 
                         (0, 255, 0), 3)
            
            cv2.putText(frame, f"Class: {class_name} - {count}/{images_per_class}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Batch Capture', frame)
            
            # Auto-capture every few frames
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            if count < images_per_class:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{class_name}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(class_dir, filename), roi)
                count += 1
            
            key = cv2.waitKey(100) & 0xFF  # Capture every 100ms
            if key == ord('q'):
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Captured {count} images for '{class_name}'")
    
    print("\nBatch capture complete!")


if __name__ == "__main__":
    print("\nSelect capture mode:")
    print("1. Interactive mode (manual capture)")
    print("2. Batch mode (auto-capture)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        batch_capture_mode()
    else:
        create_capture_session()
