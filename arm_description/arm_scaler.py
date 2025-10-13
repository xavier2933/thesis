import xml.etree.ElementTree as ET
import re

def scale_urdf(input_file, output_file, scale_factor):
    """
    Properly scale a URDF/XACRO file for use in Unity or other simulators.
    
    Args:
        input_file: Path to input URDF/XACRO
        output_file: Path to output scaled file
        scale_factor: Linear scaling factor (e.g., 1.9)
    """
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    print(f"Scaling URDF by factor of {scale_factor}...")
    
    # 1. Scale all <origin> xyz attributes (joint positions, visual/collision origins)
    origin_count = 0
    for origin in root.iter('origin'):
        xyz = origin.attrib.get('xyz')
        if xyz:
            x, y, z = map(float, xyz.split())
            origin.attrib['xyz'] = f"{x*scale_factor} {y*scale_factor} {z*scale_factor}"
            origin_count += 1
    print(f"  Scaled {origin_count} origin elements")
    
    # 2. Scale visual and collision geometries
    geometry_count = 0
    for geometry in root.iter('geometry'):
        # Cylinders
        cylinder = geometry.find('cylinder')
        if cylinder is not None:
            if 'length' in cylinder.attrib:
                length = float(cylinder.attrib['length'])
                cylinder.attrib['length'] = str(length * scale_factor)
            if 'radius' in cylinder.attrib:
                radius = float(cylinder.attrib['radius'])
                cylinder.attrib['radius'] = str(radius * scale_factor)
            geometry_count += 1
        
        # Boxes
        box = geometry.find('box')
        if box is not None and 'size' in box.attrib:
            size = list(map(float, box.attrib['size'].split()))
            box.attrib['size'] = ' '.join(str(s * scale_factor) for s in size)
            geometry_count += 1
        
        # Spheres
        sphere = geometry.find('sphere')
        if sphere is not None and 'radius' in sphere.attrib:
            radius = float(sphere.attrib['radius'])
            sphere.attrib['radius'] = str(radius * scale_factor)
            geometry_count += 1
        
        # Meshes - add scale attribute
        mesh = geometry.find('mesh')
        if mesh is not None:
            # Add or update scale attribute
            mesh.attrib['scale'] = f"{scale_factor} {scale_factor} {scale_factor}"
            geometry_count += 1
    
    print(f"  Scaled {geometry_count} geometry elements")
    
    # 3. Scale inertial properties
    inertial_count = 0
    for inertial in root.iter('inertial'):
        # Scale center of mass origin
        origin = inertial.find('origin')
        if origin is not None:
            xyz = origin.attrib.get('xyz')
            if xyz:
                x, y, z = map(float, xyz.split())
                origin.attrib['xyz'] = f"{x*scale_factor} {y*scale_factor} {z*scale_factor}"
        
        # Scale mass (proportional to volume: scale^3)
        mass = inertial.find('mass')
        if mass is not None and 'value' in mass.attrib:
            original_mass = float(mass.attrib['value'])
            mass.attrib['value'] = str(original_mass * (scale_factor ** 3))
        
        # Scale inertia tensor (proportional to mass * length^2 = scale^5)
        inertia = inertial.find('inertia')
        if inertia is not None:
            inertia_scale = scale_factor ** 5
            for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                if attr in inertia.attrib:
                    original = float(inertia.attrib[attr])
                    inertia.attrib[attr] = str(original * inertia_scale)
        
        inertial_count += 1
    
    print(f"  Scaled {inertial_count} inertial elements")
    
    # 4. Scale joint limits (effort and velocity stay the same, but you might want to adjust)
    limit_count = 0
    for limit in root.iter('limit'):
        # Note: We don't scale effort/velocity as those depend on actuator specs
        # Only scale positional limits if they're linear (prismatic joints)
        limit_count += 1
    print(f"  Found {limit_count} joint limits (not scaled - check manually if needed)")
    
    # Write output
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"\nScaled URDF written to: {output_file}")
    print(f"\nIMPORTANT NOTES:")
    print(f"  - Mesh files (.stl, .dae) are scaled via 'scale' attribute")
    print(f"  - Mass scaled by {scale_factor**3:.2f}x (volume)")
    print(f"  - Inertia scaled by {scale_factor**5:.2f}x (mass × length²)")
    print(f"  - Joint efforts/velocities NOT scaled - adjust manually if needed")


if __name__ == "__main__":
    # Example usage
    input_file = "config/panda.urdf"  # or panda.xacro - use the MAIN robot description
    output_file = "config/panda_scaled_1.9.urdf"
    scale = 1.9
    
    scale_urdf(input_file, output_file, scale)