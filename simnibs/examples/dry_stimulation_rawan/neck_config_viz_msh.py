import gmsh
from pathlib import Path

# Define the settings to add or update
simu_neck_configs_settings = {
    "General.RotationX": "273.2154637494297",
    "General.RotationY": "359.9986382488571",
    "General.RotationZ": "137.6879704689442",
    "General.ScaleX": "1.176978981525465",
    "General.ScaleY": "1.176978981525465",
    "General.ScaleZ": "1.176978981525465",
    "General.TrackballQuaternion0": "0.2479485457175294",
    "General.TrackballQuaternion1": "-0.6406840336301801",
    "General.TrackballQuaternion2": "-0.6776937782952772",
    "General.TrackballQuaternion3": "0.2622529896641705",
    "General.TranslationX": "-11.32520892547385",
    "General.TranslationY": "9.040578749816785",
    "Geometry.Clip": "0",
    "General.Clip0A": "1",
    "General.Clip0B": "0",
    "General.Clip0C": "0",
    "General.Clip0D": "0",
    "General.Clip1A": "0",
    "General.Clip1B": "1",
    "General.Clip1C": "0",
    "General.Clip1D": "0",
    "General.Clip2A": "0",
    "General.Clip2B": "0",
    "General.Clip2C": "1",
    "General.Clip2D": "0",
    "General.Clip3A": "-1",
    "General.Clip3B": "0",
    "General.Clip3C": "0",
    "General.Clip3D": "1",
    "General.Clip4A": "0",
    "General.Clip4B": "-1",
    "General.Clip4C": "0.2",
    "General.Clip4D": "20",
    "General.Clip5A": "0",
    "General.Clip5B": "0",
    "General.Clip5C": "-1",
    "General.Clip5D": "1",
    "General.ClipFactor": "5",
    "General.ClipOnlyDrawIntersectingVolume": "0",
    "General.ClipOnlyVolume": "0",
    "General.ClipPositionX": "650",
    "General.ClipPositionY": "150",
    "General.ClipWholeElements": "0",
    "Mesh.SurfaceEdges": "0",
    "Mesh.SurfaceFaces": "0",
    "Mesh.VolumeEdges": "0",
    "Mesh.VolumeFaces": "0",
    "Mesh.Clip": "1",
    "View[3].CustomMax": "10",
    "View[0].Clip": "1",
    "View[1].Clip": "1",
    "View[2].Clip": "1",
    "View[3].Clip": "16",
    "View[4].Clip": "1",
    "View[5].Clip": "1",
    "View[6].Clip": "16",
    "View[7].Clip": "16",
    "View[8].Clip": "16",
    "View[0].Visible": "0",
    "View[1].Visible": "0",
    "View[2].Visible": "0",
    "View[3].Visible": "1",
    "View[4].Visible": "0",
    "View[5].Visible": "0",
    "View[6].Visible": "0",
    "View[7].Visible": "0",
    "View[8].Visible": "1"
}


def update_opt_file(file_path):
    """Update .msh.opt file with specified settings, replacing existing values if they differ."""
    try:
        # Read the file content
        copy_file_path = file_path.parent / f"original_{file_path.name}"
        if not copy_file_path.exists():
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(copy_file_path, 'w') as file:
                file.writelines(lines)
        else:
            with open(copy_file_path, 'r') as file:
                lines = file.readlines()

        # Remove 'Show' blocks and 'Hide "*";' lines first
        in_show_block = False
        filtered_lines = []

        for line in lines:
            if line.strip().startswith("Show {"):
                in_show_block = True
                continue
            if in_show_block and line.strip() == "}":
                in_show_block = False
                continue
            if line.strip() == 'Hide "*";':
                continue
            if not in_show_block:
                filtered_lines.append(line)

        # Process filtered lines to apply new settings
        updated_lines = []
        existing_keys = {line.split("=")[0].strip(): line for line in filtered_lines}

        for line in filtered_lines:
            key = line.split("=")[0].strip()
            if key in simu_neck_configs_settings:
                # Replace line if the value differs
                new_value = simu_neck_configs_settings[key]
                if existing_keys[key].split("=")[1].strip("; \n") != new_value:
                    updated_lines.append(f"{key} = {new_value};\n")
                else:
                    updated_lines.append(line)  # Keep existing line if value matches
            else:
                updated_lines.append(line)  # Keep line if not in simu_neck_configs_settings

        # Add any missing keys from simu_neck_configs_settings that were not in the file
        for key, value in simu_neck_configs_settings.items():
            if key not in existing_keys:
                updated_lines.append(f"{key} = {value};\n")

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

        print(f"Updated settings in {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def apply_gmsh_settings():
    """Apply visualization settings"""
    gmsh.option.setNumber("General.Trackball", 1)
    gmsh.option.setNumber("General.RotationX", 273.2154637494297)
    gmsh.option.setNumber("General.RotationY", 359.9986382488571)
    gmsh.option.setNumber("General.RotationZ", 137.6879704689442)
    gmsh.option.setNumber("General.TranslationX", -11.32520892547385)
    gmsh.option.setNumber("General.TranslationY", 9.040578749816785)

    # View settings
    for i in range(9):
        try:
            gmsh.option.setNumber(f"View[{i}].Visible", 1 if i == 3 else 0)
            if i == 3:  # magnE view
                gmsh.option.setNumber(f"View[{i}].CustomMax", 10.0)
                gmsh.option.setNumber(f"View[{i}].CustomMin", 0.0)
                gmsh.option.setNumber(f"View[{i}].RangeType", 2)
                gmsh.option.setNumber(f"View[{i}].ColormapNumber", 2)
                gmsh.option.setNumber(f"View[{i}].ShowScale", 1)
        except:
            continue

    # Mesh settings
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0)
    gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)


def process_mesh_file(mesh_file, output_dir):
    """Process mesh file and export visualization"""
    # try:
    gmsh.initialize()

    print(f"Processing {mesh_file}")
    gmsh.open(str(mesh_file))

    # Apply settings
    # apply_gmsh_settings()

    # Initialize GUI context for export
    if not gmsh.fltk.isAvailable():
        gmsh.fltk.initialize()

    gmsh.fltk.wait()

    # Export image
    config_name = mesh_file.parent.name
    output_file = str(output_dir / f"{config_name}_magnE.jpg")
    gmsh.write(output_file)
    print(f"Exported: {output_file}")

    # Close GUI
    gmsh.fltk.finalize()
    gmsh.clear()




def process_directory(base_dir):
    """Process all simulation directories"""
    base_path = Path(base_dir)
    output_dir = Path(f"visualizations_{base_path.name}")
    output_dir.mkdir(exist_ok=True)

    for config_dir in base_path.glob("simu_neck_config_*"):
        if config_dir.is_dir():
            msh_files = list(config_dir.glob("*scalar.msh"))
            if msh_files:
                opt_file = msh_files[0].parent / f"{msh_files[0].name}.opt"
                if opt_file.exists():
                    update_opt_file(opt_file)
                process_mesh_file(msh_files[0], output_dir)
    if gmsh.isInitialized():
        gmsh.finalize()


if __name__ == "__main__":
    process_directory("simu_neck_configs")
