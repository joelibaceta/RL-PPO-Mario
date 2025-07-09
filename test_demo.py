#!/usr/bin/env python3
"""
Demo script showing the training validation functionality.
This script can be used to verify that the training tests work correctly.
"""

import os
import tempfile
import shutil
from tests.test_training import TestMarioTraining

def main():
    """Run a quick demonstration of the training validation tests."""
    print("🎮 Mario RL Training Validation Demo")
    print("=" * 50)
    
    # Create a temporary test instance
    test_instance = TestMarioTraining()
    
    # Setup temporary directories
    temp_dir = tempfile.mkdtemp()
    test_instance.log_dir = os.path.join(temp_dir, "logs")
    test_instance.model_dir = os.path.join(temp_dir, "models")
    os.makedirs(test_instance.log_dir, exist_ok=True)
    os.makedirs(test_instance.model_dir, exist_ok=True)
    
    try:
        print("\n1. Testing environment creation...")
        test_instance.test_environment_creation()
        print("✅ Environment creation test passed!")
        
        print("\n2. Testing trainer initialization...")
        test_instance.test_trainer_initialization()
        print("✅ Trainer initialization test passed!")
        
        print("\n3. Testing training progression (this may take a minute)...")
        test_instance.test_training_progression_with_multiple_phases()
        print("✅ Training progression test passed!")
        print("   📈 Confirmed that mean reward and episode length increase through episodes!")
        
        print("\n🎉 All training validation tests passed successfully!")
        print("\nThe tests demonstrate that:")
        print("- ✅ The model trains without errors")
        print("- ✅ Mean episode length increases through training")
        print("- ✅ Mean reward increases through training")
        print("- ✅ Model state changes during training (learning occurs)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0

if __name__ == "__main__":
    exit(main())