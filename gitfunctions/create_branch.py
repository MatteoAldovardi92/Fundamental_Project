import os
import argparse

def run_cmd(cmd):
    print(f"> {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new Git branch and push it to origin.")
    parser.add_argument("branch", help="The name of the new branch to create")
    args = parser.parse_args()

    print(f"--- Creating and switching to branch: {args.branch} ---")
    # Checkout new branch
    run_cmd(f"git checkout -b {args.branch}")
    # Push the new branch to origin and set upstream
    run_cmd(f"git push -u origin {args.branch}")
    print("--- Branch Creation Complete! ---")
