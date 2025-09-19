#!/usr/bin/env python3
"""
Diagnostic script to check if metrics.py is working properly
"""

import os
import sys

def check_file_exists():
    """Check if metrics.py exists"""
    if os.path.exists('metrics.py'):
        print("✅ metrics.py file exists")
        return True
    else:
        print("❌ metrics.py file not found")
        return False

def check_file_content():
    """Check if metrics.py has the right content"""
    try:
        with open('metrics.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key components
        if 'class MetricsCalculator:' in content:
            print("✅ MetricsCalculator class found")
        else:
            print("❌ MetricsCalculator class not found")
            return False
        
        if 'def compute_all_metrics' in content:
            print("✅ compute_all_metrics method found")
        else:
            print("❌ compute_all_metrics method not found")
            return False
        
        if 'def plot_reliability_diagram' in content:
            print("✅ plot_reliability_diagram method found")
        else:
            print("❌ plot_reliability_diagram method not found")
            return False
        
        # Check if file ends properly
        lines = content.strip().split('\n')
        last_line = lines[-1].strip()
        if last_line.endswith('return fig') or 'plt.tight_layout()' in last_line:
            print("✅ File appears to end properly")
        else:
            print("⚠️ File may not end properly")
            print(f"Last line: {last_line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading metrics.py: {e}")
        return False

def test_syntax():
    """Test if metrics.py has valid Python syntax"""
    try:
        import ast
        with open('metrics.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ metrics.py has valid Python syntax")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in metrics.py: {e}")
        print(f"Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error parsing metrics.py: {e}")
        return False

def test_import():
    """Test if we can import MetricsCalculator"""
    try:
        from metrics import MetricsCalculator
        print("✅ Successfully imported MetricsCalculator")
        
        # Test instantiation
        metrics_calc = MetricsCalculator()
        print("✅ Successfully created MetricsCalculator instance")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import MetricsCalculator: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks"""
    print("🔍 Diagnosing metrics.py issues...")
    print("=" * 50)
    
    checks = [
        ("File Exists", check_file_exists),
        ("File Content", check_file_content),
        ("Syntax Check", test_syntax),
        ("Import Test", test_import)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔎 {check_name}")
        print("-" * 30)
        result = check_func()
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! metrics.py should work now.")
        print("\nTry running: python test_compatibility.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nIf the file is incomplete, you may need to copy the complete")
        print("metrics.py content from the artifacts provided.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
