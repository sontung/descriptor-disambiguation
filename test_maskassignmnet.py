import h5py
import numpy as np
from pathlib import Path


def test_mask_assignment(h5_path, image_stem):
    print(f"Testing mask assignment for {image_stem}...")

    with h5py.File(h5_path, "r") as f:
        if image_stem not in f:
            print("❌ Image not found in H5.")
            return

        g = f[image_stem]
        bd_indices = g['bd_indices'][:]
        maskls = g['maskls'][:]
        globaldesc = g['globaldesc'][:]

        print(f"✅ Loaded features.")
        print(f"   -> Global Desc shape: {globaldesc.shape}")
        print(f"   -> Mask List shape: {maskls.shape}")
        print(f"   -> Indices shape: {bd_indices.shape}")

        # Test the pooling logic you implemented in trainer.py
        if globaldesc.ndim > 1 and globaldesc.shape[0] > 0:
            pooled_desc = np.mean(globaldesc, axis=0)
            print(f"✅ Mean-pooled desc shape: {pooled_desc.shape}")
        else:
            print("❌ Global desc is not a 2D matrix of regions.")


# Run the test on whatever the first image stem is in your dataset
test_mask_assignment("/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/descriptor_mask.h5",
                     "49")