import kagglehub
import os
import shutil
import sys

# Ensure the project root is in the path so we can import app.core
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.core.config import DATASET_URL, DATASET_PATH

def download_dataset():
    """
    Downloads the Amazon Products Sales Dataset using kagglehub.
    This method is more reliable for Python 3.13+.
    """
    print(f"Starting download for {DATASET_URL} using kagglehub...")
    
    # Check if file already exists
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already found at {DATASET_PATH}. Skipping download.")
        return

    try:
        # Download the latest version of the dataset
        path = kagglehub.dataset_download(DATASET_URL)
        print(f"Dataset downloaded to cache: {path}")
        
        # Look for the CSV inside the downloaded folder
        downloaded_files = os.listdir(path)
        csv_file = next((f for f in downloaded_files if f.endswith('.csv')), None)
        
        if csv_file:
            source_path = os.path.join(path, csv_file)
            # Copy to local directory for easier access by agents
            shutil.copy(source_path, DATASET_PATH)
            print(f"Successfully copied dataset to {DATASET_PATH}")
        else:
            print("Error: Could not find CSV file in the downloaded folder.")
            print(f"Files found: {downloaded_files}")
            
    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("\n[TIP] Ensure your kaggle.json is correctly placed in ~/.kaggle/kaggle.json")

if __name__ == "__main__":
    download_dataset()
