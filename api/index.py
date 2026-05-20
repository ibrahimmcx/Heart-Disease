import os
import sys

# Ensure root directory and src directory are in Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "src"))

from app.backend import app
