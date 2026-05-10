import os
import argparse

def run_cmd(cmd):
    print(f"> {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commit and push changes to GitHub.")
    # Message is an optional positional argument
    parser.add_argument("message", nargs="?", default="Update from Colab", help="Commit message")
    # Branch is an optional named argument defaulting to 'main'
    parser.add_argument("--branch", "-b", default="main", help="Target branch to push to (default: main)")
    args = parser.parse_args()

    print("--- Starting Update Process ---")
    run_cmd("git add .")
    run_cmd(f'git commit -m "{args.message}"')
    
    print(f"Pushing to branch: {args.branch}")
    run_cmd(f"git push origin {args.branch}")
    print("--- Update Complete! ---")
