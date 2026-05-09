import sys
import subprocess

def run_cmd(cmd):
    print(f"> {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

# Get branch name from arguments or use a default
branch_name = sys.argv[1] if len(sys.argv) > 1 else "matteo_branch"

print(f"\n--- Starting Pull Process for branch: {branch_name} ---")
run_cmd("git fetch origin")
run_cmd(f"git checkout {branch_name}")
run_cmd(f"git pull origin {branch_name}")
print("--- Pull Complete! ---\n")