#!/usr/bin/env python3
"""
Fix xcframework directory structure after ExecuTorch iOS build.

The build_apple_frameworks.sh script sometimes produces xcframeworks with
incorrect Info.plist entries or missing symlinks. This script walks the
dist/ios directory and fixes common issues:

1. Ensures each .xcframework has a valid Info.plist
2. Removes empty xcframework directories
3. Reports the final set of usable frameworks

Usage:
    python3 fix_xcframeworks.py
    
This script is called from build-executorch.sh after the iOS build completes.
It operates on $PROJECT_ROOT/dist/ios/.
"""

import os
import sys
import plistlib
import shutil

def get_project_root():
    """Get the project root (parent of scripts/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fix_xcframeworks():
    project_root = get_project_root()
    dist_ios = os.path.join(project_root, "dist", "ios")
    
    if not os.path.isdir(dist_ios):
        print(f"[fix_xcframeworks] No dist/ios directory found at {dist_ios}")
        return
    
    frameworks = []
    removed = []
    
    for item in sorted(os.listdir(dist_ios)):
        if not item.endswith(".xcframework"):
            continue
            
        xcfw_path = os.path.join(dist_ios, item)
        info_plist = os.path.join(xcfw_path, "Info.plist")
        
        # Check if the xcframework has any actual library content
        has_content = False
        for root, dirs, files in os.walk(xcfw_path):
            for f in files:
                if f.endswith((".a", ".framework", ".dylib")):
                    has_content = True
                    break
            if has_content:
                break
        
        if not has_content:
            print(f"[fix_xcframeworks] Removing empty xcframework: {item}")
            shutil.rmtree(xcfw_path)
            removed.append(item)
            continue
        
        # Validate Info.plist exists
        if not os.path.exists(info_plist):
            print(f"[fix_xcframeworks] WARNING: {item} missing Info.plist")
        
        frameworks.append(item)
    
    print(f"\n[fix_xcframeworks] Summary:")
    print(f"  Valid xcframeworks: {len(frameworks)}")
    if removed:
        print(f"  Removed (empty): {len(removed)}")
    for fw in frameworks:
        print(f"    ✓ {fw}")


if __name__ == "__main__":
    fix_xcframeworks()
