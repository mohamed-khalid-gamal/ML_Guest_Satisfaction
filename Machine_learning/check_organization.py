"""
Project Organization Summary Script

This script provides a summary of the organized project structure
and validates that all necessary files are in place.
"""

import os
from pathlib import Path
import sys

def check_file_exists(filepath):
    """Check if a file exists and return status."""
    if Path(filepath).exists():
        return "✓"
    else:
        return "✗"

def get_file_size(filepath):
    """Get file size in a human-readable format."""
    try:
        size = Path(filepath).stat().st_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024**2:
            return f"{size/1024:.1f} KB"
        elif size < 1024**3:
            return f"{size/1024**2:.1f} MB"
        else:
            return f"{size/1024**3:.1f} GB"
    except:
        return "N/A"

def main():
    """Main function to check project organization."""
    print("="*60)
    print("GUEST SATISFACTION PREDICTION - PROJECT ORGANIZATION")
    print("="*60)
    
    # Define expected structure
    structure = {
        "Root Files": [
            "README.md",
            "requirements.txt",
            "setup.py",
            "LICENSE",
            "CONTRIBUTING.md",
            "QUICKSTART.md",
            ".gitignore"
        ],
        "Source Code (src/)": [
            "src/gui.py",
            "src/preprocessing.py",
            "src/ML_Project.py",
            "src/config.py",
            "src/utils.py",
            "src/data_preprocessing.py",
            "src/model_manager.py"
        ],
        "Data Directory (data/)": [
            "data/",
        ],
        "Models Directory (models/)": [
            "models/README.md",
            "models/create_sample_model.py"
        ],
        "Notebooks Directory (notebooks/)": [
            "notebooks/",
        ],
        "Tests Directory (tests/)": [
            "tests/conftest.py",
            "tests/test_preprocessing.py"
        ],
        "Documentation (docs/)": [
            "docs/",
        ],
        "Assets (assets/)": [
            "assets/",
        ]
    }
    
    total_files = 0
    existing_files = 0
    
    for category, files in structure.items():
        print(f"\n{category}:")
        print("-" * len(category))
        
        for file in files:
            status = check_file_exists(file)
            size = get_file_size(file) if status == "✓" else ""
            
            if file.endswith("/"):
                # Directory
                if Path(file).is_dir():
                    file_count = len(list(Path(file).rglob("*")))
                    print(f"  {status} {file:<30} ({file_count} files)")
                else:
                    print(f"  {status} {file:<30} (directory)")
            else:
                # File
                print(f"  {status} {file:<30} {size}")
            
            total_files += 1
            if status == "✓":
                existing_files += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total items checked: {total_files}")
    print(f"Existing items: {existing_files}")
    print(f"Missing items: {total_files - existing_files}")
    print(f"Organization score: {existing_files/total_files*100:.1f}%")
    
    if existing_files == total_files:
        print("\n🎉 Project is perfectly organized and ready for GitHub!")
    else:
        print("\n⚠️  Some files are missing. Please check the missing items above.")
    
    print("\nRECOMMENDATIONS:")
    print("- Add your actual dataset files to the data/ directory")
    print("- Train and save models to the models/ directory")
    print("- Move Jupyter notebooks to the notebooks/ directory")
    print("- Add project documentation to the docs/ directory")
    print("- Add visualizations and plots to the assets/ directory")
    
    print("\nNEXT STEPS:")
    print("1. Initialize git repository: git init")
    print("2. Add files to git: git add .")
    print("3. Create initial commit: git commit -m 'Initial project setup'")
    print("4. Create GitHub repository")
    print("5. Push to GitHub: git push -u origin main")
    
    print("\nREADY FOR GITHUB! 🚀")

if __name__ == "__main__":
    main()
