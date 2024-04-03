import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import os
import torch
import mdtraj

def dict_coord(tr):
    """
    The `dict_coord` function takes a dictionary containing
    molecular trajectories and returns a new dictionary where
    each key is associated with the coordinates of the
    corresponding alpha carbon (CA) atoms in its trajectory.
    """
    coord={}
    for key in tr.keys():
        ca_ids = tr[key].topology.select("name CA")
        ca_xyz = tr[key].xyz[:,ca_ids]
        coord[key]=ca_xyz
    return coord



def get_distance_matrix(xyz):
    """
    Gets an ensemble of xyz conformations with shape (N, L, 3) and
    returns the corresponding distance matrices with shape (N, L, L).
    """
    return np.sqrt(np.sum(np.square(xyz[:,None,:,:]-xyz[:,:,None,:]), axis=3))

def calculate_distance_matrix_dict(xyz_dict):
       """
    Gets an ensemble of xyz conformations with shape (N, L, 3) for each key in the input dictionary
    and returns a new dictionary with the corresponding distance matrices for each key,
    with shape (N, L, L).
    """
       distance_matrix_dict = {}
       for key, xyz in xyz_dict.items():
           distance_matrix_dict[key] = get_distance_matrix(xyz)
       return distance_matrix_dict

def get_contact_map_dict(dmap_dict, threshold=0.8, pseudo_count=0.01):
        """
    Gets a dictionary of distance maps with shape (N, L, L) for each key and
    returns a new dictionary with the corresponding contact probability maps for each key,
    with shape (L, L).
    """
        contact_map_dict = {}
        for key, dmap in dmap_dict.items():
            n = dmap.shape[0]
            cmap = ((dmap <= threshold).astype(int).sum(axis=0) + pseudo_count) / n
            contact_map_dict[key] = cmap
        return contact_map_dict
    
def flatten_matrices(dictionary):
    """This function iterates through each key and matrix in the input dictionary.
      For each matrix, it applies the `flatten()` function to flatten the matrix 
      into a 1D array and saves the flattened array in the new dictionary with the same key.
        Finally, it returns the new dictionary with the flattened distributions."""
    flattened_dict = {}
    for key, matrix in dictionary.items():
        flattened_dict[key] = matrix.flatten()
    return flattened_dict


def compute_side_center_mass_dict(traj):
    """
    Returns the approximate center of mass of each side chain in the ensemble for each trajectory in the dictionary.
    Glycine residues are approximated by the coordinate of its CA atom.

    This function takes a dictionary of trajectories as input, where each trajectory represents the coordinates of atoms
    in a protein. It calculates the approximate center of mass for each side chain in the ensemble across all frames
    of each trajectory. Glycine residues are handled by using the coordinate of their CA atom.

    Parameters:
        traj_dict (dict): A dictionary where keys represent protein IDs and values are trajectory objects.

    Returns:
        center_of_mass_dict (dict): A dictionary with the same keys as the input dictionary (protein IDs).
                                     The values are arrays representing the approximate center of mass of each side chain
                                     in the ensemble for each trajectory.
    """
    center_of_mass_dict = {}

    for protein_id, traj in traj.items():
        topology = traj.topology
        n_frames = traj.n_frames
        n_residues = topology.n_residues

        scmass_ensemble = np.zeros((n_frames, n_residues, 3))

        for frame in range(n_frames):
            scmass = np.zeros((n_residues, 3))

            for i in range(n_residues):
                selection = topology.select("(resid %d) and sidechain" % i)
                if len(selection) < 1:
                    selection = topology.select("(resid %d) and (name CA)" % i)

                mass = np.array([0.0, 0.0, 0.0])
                for atom in selection:
                    mass += traj.xyz[frame, atom, :]

                scmass[i] = mass / len(selection)

            scmass_ensemble[frame] = scmass

        center_of_mass_dict[protein_id] = scmass_ensemble

    return center_of_mass_dict

def featurize_and_split_phi_psi(traj_dict):
    """This function takes a dictionary of molecular dynamics trajectories as input,
    where the keys represent protein IDs and the values contain trajectory data.
    It computes the phi and psi dihedral angles for each trajectory and returns another
    dictionary where the keys remain protein IDs, while the values are lists 
    containing the split phi and psi angle arrays, respectively."""
    phi_psi_separated = {}
    for protein_id, traj in traj_dict.items():
        atoms = list(traj.topology.atoms)
        phi_ids, phi = mdtraj.compute_phi(traj)
        psi_ids, psi = mdtraj.compute_psi(traj)
        phi_psi = np.concatenate([phi, psi], axis=1)
        split_index = phi.shape[1]  # Assuming phi and psi have the same number of angles
        phi_array, psi_array = np.split(phi_psi, [split_index], axis=1)
        array_phi = np.delete(phi_array, -1, axis=1)
        array_psi = np.delete(psi_array, 0, axis=1)
        phi_psi_separated[protein_id] = [array_phi, array_psi]
    return phi_psi_separated

def consecutive_4_indices_of_Calpha(traj_dict):
    """
    Featurize the trajectories by creating all possible 4 consecutive indices of C-alpha atoms.

    Parameters:
        traj_dict: Dictionary containing trajectories with protein IDs as keys and trajectories as values

    Returns:
        consecutive_indices_dict: Dictionary containing consecutive indices matrix for each protein
    """
    consecutive_indices_dict = {}
    
    for protein_id, traj in traj_dict.items():
        # Get all C-alpha indices.
        ca_indices = traj.topology.select("protein and name CA")

        # Create the consecutive indices matrix
        n = len(ca_indices)
        if n < 4:
            raise ValueError("Input array must contain at least 4 indices.")
        num_rows = n - 3
        consecutive_indices_matrix = np.zeros((num_rows, 4), dtype=int)
        for i in range(num_rows):
            consecutive_indices_matrix[i] = ca_indices[i:i+4]
        
        consecutive_indices_dict[protein_id] = consecutive_indices_matrix

    return consecutive_indices_dict

def rg_calculator_dict(traj_dict):
    """Compute the radius of gyration for each trajectory in a dictionary."""
    rg = {}
    for traj_id, traj_data in traj_dict.items():
        rg[traj_id] = mdtraj.compute_rg(traj_data)
    return rg
def get_dssp_dict(traj):
    """
    Calculate DSSP data for each trajectory in a dictionary and return dictionary 
    where keys are protein names and values are DSSP arrays.
    """
    protein_dssp_data_dict = {}
    for protein_name, trajectory in traj.items():
        dssp_data = mdtraj.compute_dssp(trajectory)
        protein_dssp_data_dict[protein_name] = dssp_data
    return protein_dssp_data_dict


def get_gyration_tensors_dict(trajs):
    """The function takes a dictionary of trajectories as input and computes
      the gyration tensors for each trajectory using mdtraj function, 
      returning a dictionary where keys represent trajectory names and values
        are the corresponding gyration tensors."""
    gyration_radii_dict = {}
    for protein_id, traj in trajs.items():
        # Compute the radius of gyration using mdtraj
        gyration_radius = mdtraj.compute_gyration_tensor(traj)
        # Add the gyration radius to the dictionary
        gyration_radii_dict[protein_id] = gyration_radius
    return gyration_radii_dict

def calculate_asphericity_dict(gyration_tensors_dict):
    """Calculate the asphericity for each gyration tensor in the input dictionary."""
    asphericity_dict = {}
    for k in gyration_tensors_dict.keys():
        # Initialize an empty list for each key
        asphericity_dict[k] = []
        for tensor in gyration_tensors_dict[k]:
        # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(tensor)
        # Sort eigenvalues in ascending order
            eigenvalues.sort()
        # Calculate asphericity
            lambda_max = eigenvalues[-1]
            lambda_mid = eigenvalues[1]  # Middle eigenvalue
            lambda_min = eigenvalues[0]
            asphericity = (lambda_max - lambda_min) / (lambda_max + lambda_mid + lambda_min)
        # Append the calculated asphericity to the list for the current key
            asphericity_dict[k].append(asphericity)
    return asphericity_dict


def calculate_prolateness_dict(gyration_tensors_dict):
    """Calculate the prolateness for each gyration tensor in the input dictionary."""
    prolateness_dict = {}
    for k, tensors in gyration_tensors_dict.items():
        prolateness_dict[k] = []
        for tensor in tensors:
            eigenvalues = np.linalg.eigvals(tensor)
            eigenvalues.sort()
        # Calculate asphericity
            lambda_max = eigenvalues[-1]
            lambda_mid = eigenvalues[1]  # Middle eigenvalue
            lambda_min = eigenvalues[0]
            prolateness = (lambda_max - ((lambda_mid+lambda_min)/2)) / lambda_max
            prolateness_dict[k].append(prolateness)
    return prolateness_dict

def calculate_end_to_end_distance(trajs):
    """Calculate the end-to-end distance for each trajectory."""
    end_to_end_distances = {}

    for traj_id, traj_list in trajs.items():
        for traj in traj_list:
            # Select indices of carbon alpha atoms
            ca_indices = traj.topology.select("protein and name CA")
            # Calculate the Euclidean distance between the end coordinates
            distance = mdtraj.compute_distances(traj, [[ca_indices[0], ca_indices[-1]]])
            end_to_end_distances.setdefault(traj_id, []).extend(distance.flatten())  # Flatten and extend distances

    return end_to_end_distances

def site_specific_order_parameter(ca_xyz_dict):
    """
    Computes site-specific order parameters for a set of protein conformations.
    Parameters:
        ca_xyz_dict (dict): A dictionary where keys represent unique identifiers for proteins, 
        and values are 3D arrays containing the coordinates of alpha-carbon (CA) atoms for different conformations of the protein.
    Returns:
        dict: A dictionary where keys are the same protein identifiers provided in `ca_xyz_dict`,
        and values are one-dimensional arrays containing the site-specific order parameters computed for each residue of the protein.
    """
    computed_data = {}
    for key, ca_xyz in ca_xyz_dict.items():
        starting_matrix = np.zeros((ca_xyz.shape[0], ca_xyz.shape[1]-1, ca_xyz.shape[1]-1))
        for conformation in range(ca_xyz.shape[0]):
            for i in range(ca_xyz.shape[1]-1):
                vector_i_i_plus1 = ca_xyz[conformation][i+1] - ca_xyz[conformation][i]
                vector_i_i_plus1 /= np.linalg.norm(vector_i_i_plus1)
                for j in range(ca_xyz.shape[1]-1):
                    vector_j_j_plus1 = ca_xyz[conformation][j+1] - ca_xyz[conformation][j]
                    vector_j_j_plus1 /= np.linalg.norm(vector_j_j_plus1)
                    cos_i_j = np.dot(vector_i_i_plus1, vector_j_j_plus1)
                    starting_matrix[conformation][i][j] = cos_i_j
        mean_cos_tetha = np.mean(starting_matrix, axis=0)
        square_diff = (starting_matrix - mean_cos_tetha) ** 2
        variance_cos_tetha = np.mean(square_diff, axis=0)
        K_ij = np.array([[1 - np.sqrt(2 * variance_cos_tetha[i][j]) for j in range(variance_cos_tetha.shape[1])] for i in range(variance_cos_tetha.shape[0])])
        o_i = np.mean(K_ij, axis=1)
        computed_data[key] = o_i
    return computed_data

