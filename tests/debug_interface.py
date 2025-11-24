#!/usr/bin/env python3
"""
Debug script to examine PyO3 interface output format
"""

def debug_interface_output():
    """Debug what the PyO3 interface actually returns"""
    try:
        from rust_alns_py import RustALNSInterface # type: ignore
        
        interface = RustALNSInterface()
        
        print("🔍 Examining PyO3 interface output format...")
        
        # Initialize
        print("\n1. Initialization data:")
        init_data = interface.initialize_alns("SMALL_1", 42)
        print(f"Type: {type(init_data)}")
        print(f"Keys: {list(init_data.keys()) if hasattr(init_data, 'keys') else 'Not a dict'}")
        for key, value in init_data.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        # Iteration data
        print("\n2. Iteration execution data:")
        iter_data = interface.execute_iteration(
            0,
            destroy_operator_idx=0,
            repair_operator_idx=0,
            improvement_operator_idx=None,
        )
        print(f"Type: {type(iter_data)}")
        print(f"Keys: {list(iter_data.keys()) if hasattr(iter_data, 'keys') else 'Not a dict'}")
        for key, value in iter_data.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        # Metrics data
        print("\n3. Solution metrics data:")
        metrics_data = interface.extract_solution_metrics()
        print(f"Type: {type(metrics_data)}")
        print(f"Keys: {list(metrics_data.keys()) if hasattr(metrics_data, 'keys') else 'Not a dict'}")
        for key, value in metrics_data.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        print("\n✅ Debug completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_interface_output()