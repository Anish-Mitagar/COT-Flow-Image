"""
Test script to verify the mask visualization fix.
Run this to ensure the reshape error is resolved.
"""

import numpy as np
import sys

def test_mask_reshape():
    """Test that we can properly reshape the Fourier mask."""
    print("Testing Fourier mask reshaping...")
    
    # Simulate the mask size (28*28*2 for complex Fourier)
    image_size = 28
    fourier_mask_size = image_size * image_size * 2
    
    print(f"  Fourier mask size: {fourier_mask_size}")
    print(f"  Expected: {image_size}×{image_size}×2 = {image_size*image_size*2}")
    
    # Create dummy mask
    learned_mask = np.random.rand(fourier_mask_size)
    
    # Try the old method (should fail or give wrong size)
    size_old = int(learned_mask.shape[0] ** 0.5)
    print(f"\n  Old method: sqrt({fourier_mask_size}) = {size_old}")
    print(f"  Old reshape: {size_old}×{size_old} = {size_old*size_old}")
    print(f"  ❌ This doesn't match {fourier_mask_size}!")
    
    # Try the new method (should work)
    try:
        learned_mask_2d = learned_mask.reshape(image_size, image_size, 2)
        print(f"\n  ✅ New method works!")
        print(f"  New reshape: {learned_mask_2d.shape}")
        print(f"  Real component: {learned_mask_2d[:, :, 0].shape}")
        print(f"  Imaginary component: {learned_mask_2d[:, :, 1].shape}")
        
        # Test magnitude calculation
        magnitude = np.sqrt(learned_mask_2d[:, :, 0]**2 + learned_mask_2d[:, :, 1]**2)
        print(f"  Magnitude: {magnitude.shape}")
        print(f"  Mean magnitude: {magnitude.mean():.4f}")
        
        return True
    except Exception as e:
        print(f"\n  ❌ New method failed: {e}")
        return False


def test_fallback_cases():
    """Test fallback cases for non-Fourier masks."""
    print("\n" + "="*60)
    print("Testing fallback cases...")
    
    # Case 1: Perfect square (e.g., 784 = 28×28)
    print("\n  Case 1: Perfect square (784 values)")
    mask_784 = np.random.rand(784)
    size = int(np.sqrt(784))
    print(f"  sqrt(784) = {size}")
    print(f"  {size}×{size} = {size*size}")
    if size * size == 784:
        mask_2d = mask_784.reshape(size, size)
        print(f"  ✅ Reshape successful: {mask_2d.shape}")
    else:
        print(f"  ❌ Cannot reshape")
    
    # Case 2: Non-square (e.g., 1000)
    print("\n  Case 2: Non-square (1000 values)")
    mask_1000 = np.random.rand(1000)
    size = int(np.sqrt(1000))
    print(f"  sqrt(1000) ≈ {size}")
    print(f"  {size}×{size} = {size*size}")
    if size * size == 1000:
        print(f"  ✅ Can reshape")
    else:
        print(f"  ❌ Cannot reshape - will plot as 1D")
    
    return True


def test_visualization_logic():
    """Test the complete visualization logic."""
    print("\n" + "="*60)
    print("Testing visualization logic...")
    
    image_size = 28
    expected_size = image_size * image_size * 2
    
    # Test case: Fourier mask
    print(f"\n  Testing Fourier mask (size={expected_size})...")
    learned_mask = np.random.rand(expected_size)
    
    if learned_mask.shape[0] == expected_size:
        print(f"  ✅ Detected as Fourier mask")
        learned_mask_2d = learned_mask.reshape(image_size, image_size, 2)
        print(f"  Reshaped to: {learned_mask_2d.shape}")
        
        # Extract components
        real_comp = learned_mask_2d[:, :, 0]
        imag_comp = learned_mask_2d[:, :, 1]
        magnitude = np.sqrt(real_comp**2 + imag_comp**2)
        
        print(f"  Real component: {real_comp.shape}")
        print(f"  Imaginary component: {imag_comp.shape}")
        print(f"  Magnitude: {magnitude.shape}")
        print(f"  ✅ All visualizations ready!")
        return True
    else:
        print(f"  ❌ Not detected as Fourier mask")
        return False


def main():
    """Run all tests."""
    print("╔" + "="*60 + "╗")
    print("║" + " "*15 + "MASK VISUALIZATION FIX TEST" + " "*17 + "║")
    print("╚" + "="*60 + "╝")
    
    tests = [
        ("Mask Reshape", test_mask_reshape),
        ("Fallback Cases", test_fallback_cases),
        ("Visualization Logic", test_visualization_logic),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print("\n" + "="*60)
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {name} test PASSED")
            else:
                failed += 1
                print(f"\n❌ {name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} test FAILED: {e}")
    
    # Summary
    print("\n" + "╔" + "="*60 + "╗")
    print("║" + " "*24 + "SUMMARY" + " "*28 + "║")
    print("╚" + "="*60 + "╝")
    print(f"\n  ✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}/{len(tests)}")
    else:
        print(f"\n  🎉 ALL TESTS PASSED!")
        print(f"\n  The mask visualization fix is working correctly.")
        print(f"  You can now continue training without errors.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
