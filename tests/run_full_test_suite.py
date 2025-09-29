#!/usr/bin/env python3
"""
Master Test Runner for RL-ALNS Integration Testing
Runs all test phases in sequence using SMALL_1 instance
"""

import sys
import time
import subprocess
from pathlib import Path

def main():
    """Run all test phases in sequence"""
    
    print("🚀 RL-ALNS Integration Test Suite")
    print("=" * 60)
    print("Testing with SMALL_1 instance")
    print("=" * 60)
    
    # Check if PyO3 extension is built
    print("\n🔍 Checking PyO3 extension...")
    try:
        from rust_alns_py import RustALNSInterface  # type: ignore
        print("✅ PyO3 extension found")
    except ImportError:
        print("❌ PyO3 extension not found")
        print("📋 Building PyO3 extension...")
        try:
            result = subprocess.run(
                ["maturin", "develop", "--release"],
                cwd="rust_alns",
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode == 0:
                print("✅ PyO3 extension built successfully")
            else:
                print(f"❌ Failed to build PyO3 extension:")
                print(result.stderr)
                return False
        except subprocess.TimeoutExpired:
            print("❌ Build timeout - try building manually")
            return False
        except FileNotFoundError:
            print("❌ maturin not found - install with: pip install maturin")
            return False
    
    # Test phases
    phases = [
        ("Phase 1: PyO3 Interface", "python tests/test_pyo3_interface.py", "tests/test_pyo3_interface.py"),
        ("Phase 2: RL Environment", "python tests/test_rl_environment.py", "tests/test_rl_environment.py"),
        ("Phase 3: Comprehensive Validation", "python tests/test_validation.py", "tests/test_validation.py"),
        ("Phase 4: Random Policy", "python tests/integration/test_random_policy.py", "tests/integration/test_random_policy.py"),
        ("Phase 5: PPO Training", "python tests/integration/test_ppo_training.py", "tests/integration/test_ppo_training.py"),
    ]
    
    results = {}
    total_start = time.time()
    
    for phase_name, command, script_file in phases:
        print(f"\n{'='*60}")
        print(f"🧪 {phase_name}")
        print(f"{'='*60}")
        
        # Check if test file exists
        if not Path(script_file).exists():
            print(f"❌ Test file {script_file} not found")
            results[phase_name] = "❌ SKIP: File not found"
            continue
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per phase
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print("✅ PASSED")
                results[phase_name] = f"✅ PASS ({elapsed:.1f}s)"
                
                # Show last few lines of output
                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) > 3:
                    print("📊 Test Output (last 3 lines):")
                    for line in output_lines[-3:]:
                        print(f"   {line}")
            else:
                print("❌ FAILED")
                results[phase_name] = f"❌ FAIL ({elapsed:.1f}s)"
                
                # Show error output
                if result.stderr:
                    print("🚨 Error Output:")
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-5:]:  # Last 5 lines
                        print(f"   {line}")
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"❌ TIMEOUT after {elapsed:.1f}s")
            results[phase_name] = f"❌ TIMEOUT ({elapsed:.1f}s)"
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ ERROR: {e}")
            results[phase_name] = f"❌ ERROR ({elapsed:.1f}s)"
    
    # Final results summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for result in results.values() if "✅" in result)
    total_count = len(results)
    
    for phase_name, result in results.items():
        print(f"{phase_name:<40} {result}")
    
    print(f"\n🎯 OVERALL RESULT: {passed_count}/{total_count} phases passed")
    print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ RL-ALNS integration is ready for production!")
        print("\n📋 Next steps:")
        print("   - Run larger instances")
        print("   - Tune hyperparameters")
        print("   - Implement custom RL algorithms")
    else:
        print(f"\n❌ {total_count - passed_count} phases failed")
        print("📋 Check individual test outputs above for details")
    
    return passed_count == total_count

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ("gymnasium", "gym environment framework"),
        ("numpy", "numerical computing"),
        ("stable_baselines3", "RL algorithms (optional)"),
        ("matplotlib", "plotting (optional)"),
        ("maturin", "PyO3 build tool"),
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        if "stable_baselines3" in missing_packages:
            print(f"   pip install gymnasium stable-baselines3[extra] matplotlib maturin")
        else:
            print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Pre-flight check...")
    if not check_dependencies():
        print("❌ Missing dependencies - install them first")
        sys.exit(1)
    
    print("\n🚀 Starting test suite...")
    success = main()
    
    sys.exit(0 if success else 1)