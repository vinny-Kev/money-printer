#!/usr/bin/env python3
import subprocess
import sys

try:
    result = subprocess.run([sys.executable, "comprehensive_test.py"], 
                          capture_output=True, text=True, encoding='utf-8')
    
    # Filter and print relevant lines
    lines = result.stdout.split('\n')
    for line in lines:
        if any(keyword in line for keyword in ['PASS', 'FAIL', 'Success Rate', 'FAILED TESTS', 'Duration', 'Total Tests']):
            print(line)
            
    if result.returncode != 0:
        print(f"Exit code: {result.returncode}")
        
except Exception as e:
    print(f"Error running test: {e}")
