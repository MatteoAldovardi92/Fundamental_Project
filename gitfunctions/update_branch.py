import sys
import subprocess

def run_cmd(cmd):
    print(f"> {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

# Get commit message from arguments or use a default
commit_message = sys.argv[1] if len(sys.argv) > 1 else "Auto-update from Colab"
branch_name = "matteo_branch"

print("\n--- Starting Update Process ---")
run_cmd("git add .")
run_cmd(f'git commit -m "{commit_message}"')
run_cmd(f"git push origin {branch_name}")
print("--- Update Complete! ---\n")