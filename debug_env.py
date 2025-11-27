import sys
import os
import pkg_resources

print(f"Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    import plotly
    print(f"Plotly Version: {plotly.__version__} at {plotly.__file__}")
except ImportError as e:
    print(f"Plotly Import Failed: {e}")

try:
    import streamlit
    print(f"Streamlit Version: {streamlit.__version__} at {streamlit.__file__}")
except ImportError as e:
    print(f"Streamlit Import Failed: {e}")

print("\nInstalled Packages:")
for dist in pkg_resources.working_set:
    print(f"{dist.project_name} ({dist.version})")
