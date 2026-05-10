import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"> {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    commit_message = "Update from Colab"
    if len(sys.argv) > 1:
        commit_message = sys.argv[1]
        
    print("--- Starting Update Process ---")
    run_cmd("git add .")
    run_cmd(f'git commit -m "{commit_message}"')
    
    # Automatically get the current active branch
    current_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
    
    print(f"Pushing to branch: {current_branch}")
    run_cmd(f"git push origin {current_branch}")
    print("--- Update Complete! ---")
