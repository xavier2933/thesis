#!/usr/bin/env python3
"""
Create a MoveIt config package for the scaled Panda robot
This copies the original config and replaces the URDF with your scaled version

Usage: python create_scaled_moveit_config.py
"""

import os
import shutil
from pathlib import Path

def create_scaled_moveit_config():
    # Paths
    workspace = Path.home() / "thesis_ws" / "src"
    original_pkg = Path("/opt/ros/humble/share/moveit_resources_panda_moveit_config")
    new_pkg = workspace / "panda_scaled_moveit_config"
    scaled_urdf = workspace / "arm_description" / "config" / "panda_scaled_1.9.urdf"
    
    # The SRDF is already in the original package - we'll just use that one!
    original_srdf = original_pkg / "config" / "panda.srdf"
    
    # Check if files exist
    if not scaled_urdf.exists():
        print(f"ERROR: Scaled URDF not found at {scaled_urdf}")
        return
    
    if not original_srdf.exists():
        print(f"ERROR: Original SRDF not found at {original_srdf}")
        print("Install it with: sudo apt install ros-humble-moveit-resources-panda-moveit-config")
        return
    
    if not original_pkg.exists():
        print(f"ERROR: Original MoveIt config not found at {original_pkg}")
        print("Install it with: sudo apt install ros-humble-moveit-resources-panda-moveit-config")
        return
    
    # Create new package directory
    print(f"Creating new MoveIt config package at: {new_pkg}")
    if new_pkg.exists():
        print(f"WARNING: {new_pkg} already exists. Removing...")
        shutil.rmtree(new_pkg)
    
    # Copy entire original package
    shutil.copytree(original_pkg, new_pkg)
    print("✓ Copied original MoveIt config")
    
    # Replace URDF with scaled version
    # First, need to handle the xacro file
    urdf_xacro = new_pkg / "config" / "panda.urdf.xacro"
    
    # Option 1: Replace the xacro with our expanded URDF
    # We'll rename it but keep .xacro extension for compatibility
    if urdf_xacro.exists():
        urdf_xacro.unlink()
    shutil.copy(scaled_urdf, urdf_xacro)
    print(f"✓ Replaced panda.urdf.xacro with scaled URDF")
    
    # SRDF is already good - it just references link/joint names
    # which don't change with scaling
    print(f"✓ Using existing SRDF (works with scaled robot)")
    
    # Update package.xml
    package_xml = new_pkg / "package.xml"
    with open(package_xml, 'r') as f:
        content = f.read()
    
    content = content.replace(
        '<name>moveit_resources_panda_moveit_config</name>',
        '<name>panda_scaled_moveit_config</name>'
    )
    content = content.replace(
        'MoveIt Resources Panda',
        'MoveIt Config for Scaled Panda (1.9x)'
    )
    
    with open(package_xml, 'w') as f:
        f.write(content)
    print("✓ Updated package.xml")
    
    # Create a minimal CMakeLists.txt if it doesn't exist
    cmake = new_pkg / "CMakeLists.txt"
    if not cmake.exists():
        cmake_content = """cmake_minimum_required(VERSION 3.8)
project(panda_scaled_moveit_config)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install all config and launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

ament_package()
"""
        with open(cmake, 'w') as f:
            f.write(cmake_content)
        print("✓ Created CMakeLists.txt")
    else:
        with open(cmake, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'project(moveit_resources_panda_moveit_config)',
            'project(panda_scaled_moveit_config)'
        )
        
        with open(cmake, 'w') as f:
            f.write(content)
        print("✓ Updated CMakeLists.txt")
    
    # Update launch file to use correct package name
    demo_launch = new_pkg / "launch" / "demo.launch.py"
    if demo_launch.exists():
        with open(demo_launch, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'moveit_resources_panda_moveit_config',
            'panda_scaled_moveit_config'
        )
        
        with open(demo_launch, 'w') as f:
            f.write(content)
        print("✓ Updated demo.launch.py")
    
    print("\n" + "="*60)
    print("SUCCESS! MoveIt config package created.")
    print("="*60)
    print("\nNext steps:")
    print("1. Build the package:")
    print("   cd ~/thesis_ws")
    print("   colcon build --packages-select panda_scaled_moveit_config")
    print("   source install/setup.bash")
    print("\n2. Test it:")
    print("   ros2 launch panda_scaled_moveit_config demo.launch.py")
    print("\n3. Use in your code:")
    print('   get_package_share_directory("panda_scaled_moveit_config")')
    print("="*60)

if __name__ == "__main__":
    create_scaled_moveit_config()