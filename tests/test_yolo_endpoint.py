import requests
import cv2
import numpy as np
import os

# Define endpoint URL
URL = "http://localhost:8000/verify-visual"

def create_dummy_image(filename="dummy_test.jpg"):
    """Creates a simple dummy image for testing."""
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)
    # Draw a rectangle (simulating an object)
    cv2.rectangle(img, (100, 100), (400, 400), (0, 255, 0), -1)
    cv2.putText(img, 'TEST IMAGE', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(filename, img)
    return filename

def test_endpoint():
    print("🧪 Testing /verify-visual endpoint...")
    
    # Create dummy image
    img_path = create_dummy_image()
    
    if not os.path.exists(img_path):
        print("❌ Failed to create dummy image.")
        return

    try:
        # Send request
        with open(img_path, "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            print(f"   Sending {img_path} to {URL}...")
            response = requests.post(URL, files=files)
        
        # Check response
        if response.status_code == 200:
            print("✅ Request successful!")
            print("   Response:")
            print(response.json())
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running? (Run 'uvicorn src.api_service.main:app --reload')")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    test_endpoint()
