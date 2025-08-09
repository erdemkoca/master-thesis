#!/usr/bin/env python3
"""
Quick test of comprehensive experiment (3 iterations only)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Modify the comprehensive script to run just 3 iterations for testing
exec(open('03_run_experiments_synthetic_COMPREHENSIVE.py').read().replace(
    'N_ITERATIONS = 20', 'N_ITERATIONS = 3'
).replace(
    'N_TEST_LARGE = 50000', 'N_TEST_LARGE = 5000'  # Smaller for speed
))
