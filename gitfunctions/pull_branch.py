import os
import argparse

def run_cmd(cmd):
    print(f"> {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull latest changes from GitHub.")
    # Make branch an optional positional argument defaulting to 'main'
    parser.add_argument("branch", nargs="?", default="main", help="Target branch to pull (default: main)")
    args = parser.parse_args()

    print(f"--- Pulling latest changes for branch: {args.branch} ---")
    run_cmd("git fetch origin")
    run_cmd(f"git checkout {args.branch}")
    run_cmd(f"git pull origin {args.branch}")
    print("--- Pull Complete! ---")
