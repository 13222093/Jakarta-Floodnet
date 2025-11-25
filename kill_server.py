import os
import subprocess

def kill_port(port):
    try:
        cmd = f"netstat -ano | findstr :{port}"
        output = subprocess.check_output(cmd, shell=True).decode()
        lines = output.strip().split('\n')
        for line in lines:
            parts = line.split()
            pid = parts[-1]
            if pid != '0':
                print(f"Killing PID {pid}")
                os.system(f"taskkill /F /PID {pid}")
    except:
        pass

kill_port(8000)
