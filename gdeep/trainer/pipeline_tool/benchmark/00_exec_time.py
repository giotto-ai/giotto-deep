import subprocess
from pathlib import Path
import sys
import os
# Ajoutez le chemin du projet à sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

dir_path = Path(__file__).resolve().parent / "training_launcher.py"
p = subprocess.run(['python', dir_path,
                            "exec_time",
                            "CNN",
                            "API torch",
                            '--gpu', str(1),
                            '--chunk', str(2),
                            '--epochs', str(2)], capture_output=True, text=True)

result = p.stdout
print(result)
print("Starting pipeline on 1 gpu")

p = subprocess.run(['python', dir_path,
                            "exec_time",
                            "CNN",
                            "Pipeline",
                            '--gpu', str(1),
                            '--chunk', str(2),
                            '--epochs', str(2)], capture_output=True, text=True)

result = p.stdout
print(result)
print("Starting pipeline on 2 gpu")

p = subprocess.run(['python', dir_path,
                            "exec_time",
                            "CNN",
                            "Pipeline",
                            '--gpu', str(2),
                            '--chunk', str(2),
                            '--epochs', str(2)], capture_output=True, text=True)

result = p.stdout
print(result)
