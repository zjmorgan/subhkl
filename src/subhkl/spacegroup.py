import numpy as np
import gemmi

def get_space_group_object(sg_name_or_centering):
    """
    Returns a gemmi.SpaceGroup object.
    Handles legacy 'centering' chars (P, A, F, etc) by mapping to minimal space groups.
    """
    # Clean up input
    name = str(sg_name_or_centering).strip()
    
    # Handle single-letter centering codes by mapping to lowest symmetry group
    # This maintains backward compatibility with 'P', 'F', 'I', etc.
    centering_map = {
        'P': 'P 1',
        'A': 'A 1', # gemmi handles non-standard settings if provided correctly, or use P1 + centering op
        'B': 'B 1',
        'C': 'C 1',
        'I': 'I 1',
        'F': 'F 1',
        'R': 'R 3', # R lattice usually implies R3 or R3m basis
        'H': 'H 1'  # Hexagonal centering? usually just P for Hex
    }
    
    if name.upper() in centering_map:
        name = centering_map[name.upper()]
        
    try:
        sg = gemmi.SpaceGroup(name)
    except RuntimeError:
        # Fallback: try interpreting as Hall symbol or int
        try:
            sg = gemmi.SpaceGroup(int(name))
        except ValueError:
            raise ValueError(f"Could not interpret space group: {sg_name_or_centering}")
            
    return sg

def get_centering(space_group_name):
    sg = get_space_group_object(space_group_name)
    return sg.centring_type()

def is_systematically_absent(h, k, l, space_group_name):
    """
    Check systematic absences for arrays of h, k, l using Gemmi.
    """
    sg = get_space_group_object(space_group_name)
    ops = sg.operations() # <--- FIXED: Get operations object
    
    result = []
    for hi, ki, li in zip(h, k, l):
        if hi == 0 and ki == 0 and li == 0:
            result.append(True) # 000 is physically absent
            continue
            
        # Gemmi check
        is_absent = ops.is_systematically_absent([int(hi), int(ki), int(li)])
        result.append(is_absent)
        
    return np.array(result, dtype=bool)

def generate_hkl_mask(h_max, k_max, l_max, space_group_name):
    """
    Generates a dense 3D boolean mask of valid (allowed) reflections.
    Used for JAX lookups.

    Shape: (2*h_max+1, 2*k_max+1, 2*l_max+1)
    """
    sg = get_space_group_object(space_group_name)
    ops = sg.operations()

    h_range = np.arange(-h_max, h_max + 1)
    k_range = np.arange(-k_max, k_max + 1)
    l_range = np.arange(-l_max, l_max + 1)

    H, K, L = np.meshgrid(h_range, k_range, l_range, indexing='ij')

    mask = np.ones(H.shape, dtype=bool)

    # Iterate and check
    it = np.nditer([H, K, L], flags=['multi_index'])
    for h, k, l in it:
        if h == 0 and k == 0 and l == 0:
            mask[it.multi_index] = False
            continue

        if ops.is_systematically_absent([int(h), int(k), int(l)]):
            mask[it.multi_index] = False

    return mask
