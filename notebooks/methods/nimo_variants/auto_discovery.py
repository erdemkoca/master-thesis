"""
Automated NIMO Variant Discovery

This module provides functionality to automatically discover and import
all NIMO variants from the nimo_variants folder.
"""

import glob
import importlib
import pathlib
from typing import List, Callable


def discover_nimo_variants() -> List[Callable]:
    """
    Automatically discover and import all run_nimo_* variants from the nimo_variants folder.
    
    Returns:
        List of callable functions (run_nimo_* functions)
    """
    variant_funcs = []
    
    # Get the current directory (where this file is located)
    current_dir = pathlib.Path(__file__).parent
    
    # Find all Python files in the nimo_variants directory
    for path in glob.glob(str(current_dir / "*.py")):
        # Skip this file and __init__.py
        if pathlib.Path(path).name in ["auto_discovery.py", "__init__.py"]:
            continue
            
        stem = pathlib.Path(path).stem
        
        try:
            # Import the module - try different import paths
            module = None
            import_paths = [
                f"methods.nimo_variants.{stem}",
                f"nimo_variants.{stem}",
                stem
            ]
            
            for import_path in import_paths:
                try:
                    module = importlib.import_module(import_path)
                    break
                except ImportError:
                    continue
            
            if module is None:
                print(f"✗ Error: Could not import {stem}.py with any import path")
                continue
            
            # Get the function (should be named run_nimo_{stem})
            func_name = f"run_nimo_{stem}"
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                variant_funcs.append(func)
                print(f"✓ Discovered: {func_name}")
            else:
                print(f"⚠ Warning: {func_name} not found in {stem}.py")
                
        except AttributeError as e:
            print(f"✗ Error accessing function in {stem}.py: {e}")
    
    print(f"\nTotal variants discovered: {len(variant_funcs)}")
    return variant_funcs


def get_variant_names() -> List[str]:
    """
    Get the names of all discovered variants.
    
    Returns:
        List of variant names (e.g., ['baseline', 'variant'])
    """
    funcs = discover_nimo_variants()
    names = []
    
    for func in funcs:
        # Extract variant name from function name (run_nimo_baseline -> baseline)
        name = func.__name__.replace('run_nimo_', '')
        names.append(name)
    
    return names


# Example usage:
if __name__ == "__main__":
    print("Discovering NIMO variants...")
    variants = discover_nimo_variants()
    
    print("\nAvailable variants:")
    for i, func in enumerate(variants, 1):
        print(f"{i}. {func.__name__}")
    
    print(f"\nYou can now use these in your methods list:")
    print("methods = [")
    print("    run_lasso,")
    print("    run_random_forest,")
    print("    run_neural_net,")
    print("    *discover_nimo_variants()  # Auto-discover all variants")
    print("]") 