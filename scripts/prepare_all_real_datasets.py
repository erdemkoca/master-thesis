#!/usr/bin/env python3
"""
Master script to prepare all real datasets
"""

import os
import sys
import subprocess

def run_script(script_name):
    """Run a preparation script."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print(f"❌ {script_name} failed")
            if result.stderr:
                print("Error:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    
    return True

def main():
    """Prepare all real datasets."""
    print("REAL DATASET PREPARATION MASTER SCRIPT")
    print("=" * 60)
    print("This script will prepare all real datasets for the unified pipeline.")
    print("Make sure you have downloaded the raw datasets to the appropriate raw/ directories.")
    print()
    
    # List of preparation scripts
    scripts = [
        "prepare_breast_cancer.py",  # Downloads from sklearn
        "prepare_pima.py",           # Requires manual download
        "prepare_bank_marketing.py", # Requires manual download
        "prepare_credit_default.py", # Requires manual download
        "prepare_spambase.py"        # Requires manual download
    ]
    
    # Check which datasets are available
    available_scripts = []
    missing_datasets = []
    
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(script_path):
            available_scripts.append(script)
        else:
            missing_datasets.append(script)
    
    print(f"Available preparation scripts: {len(available_scripts)}")
    print(f"Missing scripts: {len(missing_datasets)}")
    
    if missing_datasets:
        print(f"\nMissing scripts: {missing_datasets}")
    
    # Run available scripts
    success_count = 0
    for script in available_scripts:
        if run_script(script):
            success_count += 1
    
    print(f"\n{'='*60}")
    print("PREPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"Scripts run: {len(available_scripts)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(available_scripts) - success_count}")
    
    if success_count == len(available_scripts):
        print("\n🎉 All available datasets prepared successfully!")
        print("You can now run the unified pipeline with both synthetic and real datasets.")
    else:
        print(f"\n⚠️  Some datasets failed to prepare. Check the errors above.")
    
    return success_count == len(available_scripts)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
