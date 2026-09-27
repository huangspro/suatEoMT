#!/usr/bin/env python3
"""
Check pairing status between image and mask directories.
Reports files that exist in one directory but not the other.
"""

import os

# Configuration

IMAGE_DIR = "../myDataset/data/leftImg8bit/train/aachen"
MASK_DIR = "../myDataset/data/gtFine/train/aachen"


# Main Function

def check_both_dirs():
    """Bidirectional comparison between image and mask directories."""
    # Get all PNG files from both directories
    images = set([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
    masks = set([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    
    # Compute set differences
    only_in_images = images - masks
    only_in_masks = masks - images
    common = images & masks
    
    # Print summary statistics
    print("=" * 60)
    print("DATASET PAIRING STATUS REPORT")
    print("=" * 60)
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Mask directory:  {MASK_DIR}")
    print("-" * 60)
    print(f"Total images:           {len(images)}")
    print(f"Total masks:            {len(masks)}")
    print(f"Common files:           {len(common)}")
    print(f"Only in images:         {len(only_in_images)}")
    print(f"Only in masks:          {len(only_in_masks)}")
    print("=" * 60)
    
    # Report files missing from images
    if only_in_masks:
        print("\n" + "!" * 60)
        print(f"FILES ONLY IN MASKS (missing from images): {len(only_in_masks)}")
        print("!" * 60)
        for f in sorted(only_in_masks)[:20]:
            print(f"  ✗ {f}")
        if len(only_in_masks) > 20:
            print(f"  ... and {len(only_in_masks) - 20} more files")
        
        # Save full list to file for further inspection
        with open("missing_in_images.txt", "w") as out:
            out.write("\n".join(sorted(only_in_masks)))
        print(f"\nFull list saved to: missing_in_images.txt")
    
    # Report files missing from masks
    if only_in_images:
        print("\n" + "!" * 60)
        print(f"FILES ONLY IN IMAGES (missing from masks): {len(only_in_images)}")
        print("!" * 60)
        for f in sorted(only_in_images)[:20]:
            print(f"  ✗ {f}")
        if len(only_in_images) > 20:
            print(f"  ... and {len(only_in_images) - 20} more files")
        
        # Save full list to file for further inspection
        with open("missing_in_masks.txt", "w") as out:
            out.write("\n".join(sorted(only_in_images)))
        print(f"\nFull list saved to: missing_in_masks.txt")
    
    # Final summary
    if not only_in_images and not only_in_masks:
        print("\n All images and masks are properly paired.")
    else:
        print("\n⚠ Dataset is not perfectly paired. Check the lists above.")


if __name__ == "__main__":
    check_both_dirs()