from copy import deepcopy
from pathlib import Path

import numpy as np
from simnibs import sim_struct, run_simnibs
from simnibs.simulation import ELECTRODE
from simnibs import ElementTags
from simnibs.utils.mesh_element_properties import tissue_conductivities
from simnibs.utils.simnibs_logger import logger

m2m_folder_path = Path(f'm2m_ernie')
results_path = Path('simu_neck_configs')

# Electrode positions
elec1_center = [-66.5, -3.35, -125.5]  # bottom
elec2_center = [-68.4, -0.4, -108.2]  # top

# Constants
SHAPE = 'shape'  # 'ellipse' or 'rect' or 'custom'
DIMENSIONS = 'dimensions'  # [width, height] or [diameter] in mm
THICKNESS = 'thickness'  # thickness
CONDUCTIVITY = 'conductivity'  # S/m
CENTRE = 'centre'  # [x, y] in mm
NAME = 'name'

# Properties of the electrodes
electrodes_thickness = [1, 1]  # thickness of the electrodes in mm [layer1-gel, layer2-rubber]
electrodes_properties = [
    # pos_ydir is not important for circle
    {
        SHAPE: 'ellipse',
        DIMENSIONS: [9, 9],
        THICKNESS: electrodes_thickness,
        CONDUCTIVITY: 0.08442 ## TODO: Automatically calculate the conductivity from the script in my main repo.
    },
    {
        SHAPE: 'ellipse',
        DIMENSIONS: [15, 15],
        THICKNESS: electrodes_thickness,
        CONDUCTIVITY: 0.030391
    },
    {
        SHAPE: 'ellipse',
        DIMENSIONS: [15, 20],
        THICKNESS: electrodes_thickness,
        CONDUCTIVITY: 0.022793
    },
]


def config_name(config):
    return f'config_{config[SHAPE]}_{config[DIMENSIONS][0]}x{config[DIMENSIONS][1]}mm_{config[CONDUCTIVITY]}Sm-1'


electrodes_properties = [{NAME: config_name(config), **config} for config in electrodes_properties]


def configure_electrode(properties):
    """
    Configure an ELECTRODE object with any given properties.
    Properties not specified will keep their default values.

    Args:
        properties (dict): Dictionary containing electrode properties to change
            Possible keys:
            - 'shape': str
            - 'dimensions': List[float]
            - 'thickness': List[float]
            - 'conductivity': float
            (and any other valid ELECTRODE properties)

    Returns:
        ELECTRODE: Configured electrode object
    """
    electrode = ELECTRODE()

    # Configure only the properties that are provided
    for prop_name, value in properties.items():
        if prop_name == CONDUCTIVITY:
            # Change the tissue conductivities
            tissue_conductivities[ElementTags.SALINE] = value  # S/m   Bottom layer
            tissue_conductivities[ElementTags.ELECTRODE_RUBBER] = 29.4  # S/m   Top layer
        elif prop_name != 'name':  # Skip the name property when configuring electrode
            setattr(electrode, prop_name, value)
    return electrode


def elec_distance(elec1_center, elec2_center):
    """Calculate the Euclidean distance between two electrode centers."""
    return np.linalg.norm(np.array(elec2_center) - np.array(elec1_center))


def create_simulation_session(subject_path, simulation_name):
    """
    Create and configure a simulation session.

    Save fields options:
        v: electric potential at the nodes
        e: Electric field magnitude at the elements
        E: Electric field vector at the elements
        j: Current density magnitude at the elements
        J: Current density vector at the elements
        s: Conductivity at the elements
        D: dA/dt at the nodes
        g: gradiet of the potential at the elements
    """
    S = sim_struct.SESSION()
    S.subpath = subject_path
    S.pathfem = simulation_name
    S.fields = 'veEjJsDg'
    S.map_to_surf = False  # Map to subject's middle gray matter surface
    S.open_in_gmsh = False  # show results in gmsh (not for the the niftis)
    S.map_to_fsavg = False
    S.map_to_MNI = False
    return S


def run_electrode_simulation(electrode_config, subject_path, results_path, elec1_center, elec2_center):
    """
    Run a single simulation with the given electrode configuration.

    Args:
        electrode_config (dict): Electrode configuration dictionary
        subject_path (str): Path to the subject's m2m folder
        elec1_center (list): Center coordinates for first electrode
        elec2_center (list): Center coordinates for second electrode
    """
    # Create base plug configuration
    plug = configure_electrode({
        SHAPE: 'ellipse',
        DIMENSIONS: [4, 4],
        CENTRE: [0, 0]
    })

    # Configure electrode with the current properties
    elec_config = configure_electrode(electrode_config)
    elec_config.add_plug(plug=plug)

    # Calculate and log electrode distance
    dist = elec_distance(elec1_center, elec2_center)
    logger.info(f"Configuration: {electrode_config['name']}")
    logger.info(f"Distance between electrodes: {dist:0.2f} mm")

    # Create simulation session
    simulation_name = f"{results_path}/simu_neck_{electrode_config['name']}"
    S = create_simulation_session(subject_path, simulation_name)

    # Configure tDCS
    tdcs = S.add_tdcslist()
    tdcs.currents = np.array([1, -1]) * 1e-3  # Current flow (A)

    # Add first electrode
    el1 = tdcs.add_electrode(electrode=deepcopy(elec_config))
    el1.channelnr = 1
    el1.centre = elec1_center

    # Add second electrode
    el2 = tdcs.add_electrode(electrode=deepcopy(elec_config))
    el2.channelnr = 2
    el2.centre = elec2_center

    # Run simulation
    run_simnibs(S)


def main():
    results_path.mkdir(exist_ok=True)

    # Run simulations for all electrode configurations
    for config in electrodes_properties:
        try:
            run_electrode_simulation(config, m2m_folder_path, results_path, elec1_center, elec2_center)
            logger.info(f"Successfully completed simulation for {config['name']}")
        except Exception as e:
            logger.error(f"Error running simulation for {config['name']}: {str(e)}")


if __name__ == "__main__":
    main()