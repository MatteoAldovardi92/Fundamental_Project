# 🛠️ Git Helper Functions (`gitfunctions/`)

*Disclaimer: This document is AI generated.*

Since this project is primarily developed inside Google Colab, standard terminal Git workflows can be cumbersome. This folder provides Python scripts to simplify interacting with GitHub.

## 📝 Key Scripts:
- **`main.ipynb`**: Your 'sadder version of GitHub Desktop!' This notebook serves as an interactive UI to run these Git helper commands directly within Colab, saving you from wrestling with terminal outputs.
- **`update_branch.py`**: A one-click script to stage all changes, commit them with a message, and push them to the active branch. 
  - *Usage:* `!python gitfunctions/update_branch.py "My message" --branch main` (defaults to main if branch is omitted).
- **`pull_branch.py`**: Quickly pulls the latest changes from the remote repository to ensure your Colab environment is up to date.
  - *Usage:* `!python gitfunctions/pull_branch.py main` (defaults to main if omitted).
- **`create_branch.py`**: Creates a new branch locally and pushes it to the GitHub repository so your colleagues can collaborate on it.
  - *Usage:* `!python gitfunctions/create_branch.py new_feature_branch`
