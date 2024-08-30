import subprocess
import sys

def start_jupyter_lab():
    try:
        subprocess.run(["jupyter", "lab"], check=True)
    except subprocess.CalledProcessError:
        print("Failed to start Jupyter Lab. Make sure it's installed correctly.")
        sys.exit(1)

if __name__ == "__main__":
    start_jupyter_lab()