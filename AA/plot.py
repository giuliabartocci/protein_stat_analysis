import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns
import mdtraj


def plot_average_dmap_comparison(dmap_ref, dmap_gen, title,
                                 ticks_fontsize=14,
                                 cbar_fontsize=14,
                                 title_fontsize=14,
                                 dpi=96,
                                 max_d=6.8,
                                 use_ylabel=True):
    plt.figure(dpi=dpi)
    if dmap_ref is not None:
        _dmap_ref = dmap_ref.mean(axis=0)
    else:
        _dmap_ref = np.zeros(dmap_gen.shape[1:])
    _dmap_gen = dmap_gen.mean(axis=0)
    tril_ids = np.tril_indices(dmap_gen.shape[2], 0)
    _dmap_ref[tril_ids[0],tril_ids[1]] = 0
    _dmap_gen[tril_ids[0],tril_ids[1]] = 0
    dmap_sel = _dmap_ref.T + _dmap_gen
    diag_ids = np.diag_indices(dmap_sel.shape[0])
    dmap_sel[diag_ids] = np.nan
    if dmap_ref is None:
        dmap_sel[tril_ids[0],tril_ids[1]] = np.nan
    plt.imshow(dmap_sel)
    plt.xticks(fontsize=ticks_fontsize)
    if use_ylabel:
        plt.yticks(fontsize=ticks_fontsize)
    else:
        plt.yticks([])
    plt.title(title, fontsize=title_fontsize)
    cbar = plt.colorbar()
    cbar.set_label(r"Average $d_{ij}$ [nm]")
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    if max_d is not None:
        plt.clim(0, max_d)
    plt.show()
    

    
    
def plot_cmap_comparison(cmap_ref, cmap_gen, title,
                         ticks_fontsize=14,
                         cbar_fontsize=14,
                         title_fontsize=14,
                         dpi=96, cmap_min=-3.5,
                         use_ylabel=True):
    plt.figure(dpi=dpi)
    cmap = cm.get_cmap("jet")
    norm = colors.Normalize(cmap_min, 0)
    if cmap_ref is not None:
        _cmap_ref = np.log10(cmap_ref.copy())
    else:
        _cmap_ref = np.zeros(cmap_gen.shape)
    _cmap_gen = np.log10(cmap_gen.copy())
    tril_ids = np.tril_indices(cmap_gen.shape[0], 0)
    _cmap_ref[tril_ids[0],tril_ids[1]] = 0
    _cmap_gen[tril_ids[0],tril_ids[1]] = 0
    cmap_sel = _cmap_ref.T + _cmap_gen
    diag_ids = np.diag_indices(cmap_sel.shape[0])
    cmap_sel[diag_ids] = np.nan
    if cmap_ref is None:
        cmap_sel[tril_ids[0],tril_ids[1]] = np.nan
    plt.imshow(cmap_sel, cmap=cmap, norm=norm)
    plt.xticks(fontsize=ticks_fontsize)
    if use_ylabel:
        plt.yticks(fontsize=ticks_fontsize)
    else:
        plt.yticks([])
    plt.title(title, fontsize=title_fontsize)
    cbar = plt.colorbar()
    cbar.set_label(r'$log_{10}(p_{ij})$')
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    plt.clim(cmap_min, 0)
    plt.show()
    
def plot_flattened_data(flattened_dict):
    """It creates a number of subplots corresponding to the number of ensembles 
    in the provided dictionary. Each subplot shows a histogram of the flattened data
      from the corresponding ensemble, with a title indicating the ensemble's name."""
    num_plots = len(flattened_dict)
    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5))

    # Customizing seaborn style
    sns.set(style="whitegrid")

    for i, (key, flattened_data) in enumerate(flattened_dict.items()):
        sns.histplot(flattened_data, bins=20, ax=axs[i], color='skyblue', edgecolor='black')
        axs[i].set_title(f'Distribution - {key}', fontsize=14)
        axs[i].set_xlabel('Value', fontsize=12)
        axs[i].set_ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_boxplot(flattened_dict):
    """Creates subplots for visualizing distributions of flattened data from different ensembles using boxplots."""
    num_plots = len(flattened_dict)
    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5))

    # Customizing seaborn style
    sns.set(style="whitegrid")

    for i, (key, flattened_data) in enumerate(flattened_dict.items()):
        sns.boxplot(y=flattened_data, ax=axs[i], color='skyblue')
        axs[i].set_title(f'Distribution - {key}', fontsize=14)
        axs[i].set_ylabel('Value', fontsize=12)
        axs[i].set_xlabel('Ensemble', fontsize=12)
    
    plt.tight_layout()
    plt.show()



def plot_distances_comparison(prot_data, polyala_data, n_residues,
                              dpi=96, n_hist=5):

    all_pairs = np.array([(i,j) for i in range(n_residues) \
                          for j in range(n_residues) if i < j])
    sel_pairs = all_pairs[np.random.choice(all_pairs.shape[0], n_hist,
                                           replace=False)]

    h_args = {"histtype": "step", "density": True}
    protein_count = 0
    bins_list = []
    _polyala_data = (("polyala", polyala_data[0], polyala_data[0]), )
    for protein_s, dmap_md_s, dmap_gen_s in prot_data + _polyala_data:
        fig, ax = plt.subplots(1, n_hist, figsize=(2.5*n_hist, 3), dpi=dpi)
        hist_l = []
        labels_l = []
        for k in range(n_hist):
            i, j = sel_pairs[k]
            # MD data.
            h_0k = ax[k].hist(dmap_md_s[:,i,j], bins=50, **h_args)
            bins_ij = h_0k[1]
            if protein_count == 0:
                bins_list.append(bins_ij)
            else:
                bins_ij = bins_list[k]
            if k == 0:
                hist_l.append(h_0k[2][0])
                labels_l.append("MD")
            # Generated data.
            h_1k = ax[k].hist(dmap_gen_s[:,i,j], bins=bins_ij, **h_args)
            if k == 0:
                hist_l.append(h_1k[2][0])
                labels_l.append("GEN")
            # Polyala data.
            if protein_s != "polyala":
                h_2k = ax[k].hist(polyala_data[0][:,i,j],
                                  bins=bins_ij, alpha=0.6,
                                  **h_args)
                if k == 0:
                    hist_l.append(h_2k[2][0])
                    labels_l.append("ALA MD")
            ax[k].set_title("residues %s-%s" % (i, j))
            if k == 0:
                ax[k].set_ylabel("density")
            ax[k].set_xlabel("distance [nm]")
        protein_count += 1
        fig.legend(hist_l, labels_l, "upper right", ncol=3)
        plt.suptitle(protein_s, fontweight='bold')
        plt.tight_layout()
        plt.show()


def plot_rg_comparison(prot_data, polyala_data, n_bins=50,
                       bins_range=(1, 4.5), dpi=96):
    h_args = {"histtype": "step", "density": True}
    n_systems = len(prot_data)+1
    bins = np.linspace(bins_range[0], bins_range[1], n_bins+1)
    fig, ax = plt.subplots(1, n_systems, figsize=(3*n_systems, 3), dpi=dpi)
    for i, (name_i, rg_md_i, rg_gen_i) in enumerate(prot_data):
        ax[i].hist(rg_md_i, bins=bins, label="MD", **h_args)
        ax[i].hist(rg_gen_i, bins=bins, label="GEN", **h_args)
        ax[i].hist(polyala_data[0], bins=bins, alpha=0.6, label="ALA MD",
                   **h_args)
        ax[i].legend()
        ax[i].set_title(name_i)
        if i == 0:
            ax[i].set_ylabel("density")
        ax[i].set_xlabel("Rg [nm]")
    ax[-1].hist(polyala_data[0], bins=bins, label="MD", **h_args)
    ax[-1].hist(polyala_data[1], bins=bins, label="GEN", **h_args)
    ax[-1].legend()
    ax[-1].set_title("polyala")
    ax[-1].set_xlabel("Rg [nm]")
    plt.tight_layout()
    plt.show()
    
    
def plot_rg_distribution(rg_vals, title, n_bins=50, dpi=96):
    plt.figure(dpi=dpi)
    plt.hist(rg_vals, bins=n_bins, histtype="step", density=True)
    plt.xlabel("Rg [nm]")
    plt.ylabel("density")
    plt.title(title)
    plt.show()
    
    
def plot_dmap_snapshots(dmap, n_snapshots, dpi=96):
    fig, ax = plt.subplots(1, n_snapshots,
                           figsize=(3*n_snapshots, 3), dpi=dpi,
                           constrained_layout=True)
    random_ids = np.random.choice(dmap.shape[0], n_snapshots,
                                  replace=dmap.shape[0] < n_snapshots)
    for i in range(n_snapshots):
        _dmap_i = dmap[random_ids[i]]
        im_i = ax[i].imshow(_dmap_i)
        ax[i].set_title("snapshot %s" % random_ids[i])
    cbar = fig.colorbar(im_i, ax=ax, location='right', shrink=0.8)
    cbar.set_label(r"$d_{ij}$ [nm]")
    plt.show()

def Ramachandran_plot_phi_psi(protein_phi_psi_dict, figsize=(10, 6), background_colour='viridis'):
    
    """ Parameters:
        protein_phi_psi_dict (dict): A dictionary where keys are protein IDs and values are tuples containing
                                     arrays of phi and psi angles.
        figsize (tuple, optional): Size of the figure (width, height) in inches. Default is (10, 6).
        background_colour (str, optional): Colour map for the plot background. Default is 'viridis'.

    Output:
       Creates a 2D histogram (Ramachandran plot) of phi and psi angles for each protein in the given dictionary.
    """
    
    for protein_id, angles in protein_phi_psi_dict.items():
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        # Remove fig.patch.set_visible(False) to show the title
        kwargs = {
            "bins": 140,
            "cmap": background_colour,
            "norm": colors.PowerNorm(0.1),
            "alpha": 1
        }
        # Flatten phi and psi arrays
        phi_flat = np.ravel(angles[0])
        psi_flat = np.ravel(angles[1])
        
        # Reference https://matplotlib.org/3.2.1/gallery/statistics/hist.html for help with making 2D histogram.
        hist = ax.hist2d(phi_flat, psi_flat, **kwargs)
        plt.colorbar(hist[3], ax=ax)  # Add colorbar
        ax.set_title(f"Ramachandran Plot for {protein_id}")  # Set the title
        ax.set_xlabel('Phi Angles (degrees)')  # Set x-axis label
        ax.set_ylabel('Psi Angles (degrees)')  # Set y-axis label
        plt.show()

def scatter_plot_phi_psi(protein_phi_psi_dict):
    """It takes in input a dictionary where keys are protein IDs and values are lists of tuples
    containing phi and psi angles and plots scatter plots of phi and psi angles for each protein.
    """
    for protein_id, phi_psi_values in protein_phi_psi_dict.items():
        phi_values =  phi_psi_values[0]
        psi_values = phi_psi_values[1]

        plt.figure(figsize=(8, 6))
        plt.scatter(phi_values, psi_values, c='b', marker='o', alpha=0.5)
        plt.xlabel('Phi Angle')
        plt.ylabel('Psi Angle')
        plt.title(f'Scatter Plot of Phi and Psi Angles for Protein {protein_id}')
        plt.grid(True)
        plt.show()

def plot_distribution_dihedral_angles(trajectories_angles, four_consecutive_indices_Calpha):
    """Parameters:
        trajectories_angles (dict): A dictionary where keys are protein IDs and values are trajectory data for each protein.
        four_consecutive_indices_Calpha (dict): A dictionary where keys are protein IDs and values are arrays of indices 
                                                representing consecutive C-alpha atoms.
        Output: Plot the distribution of dihedral angles between four consecutive C-alpha atoms for each protein.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for k in trajectories_angles.keys():
        ens_dh_ca = mdtraj.compute_dihedrals(trajectories_angles[k], four_consecutive_indices_Calpha[k]).ravel()
        ax.hist(ens_dh_ca, bins=50, histtype="step", density=True, label=f"Protein {k}")

    ax.set_title("Distribution of Dihedral Angles between Four Consecutive C-alpha Beads")
    ax.set_xlabel("Dihedral Angle [degrees]")
    ax.set_ylabel("Density")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_distribution_of_Rgs(angle_dict):
    """
    Plot the distribution of angles from a dictionary of trajectories.
    """
    plt.figure(figsize=(10, 6))
    for traj_id, angles in angle_dict.items():
        plt.hist(angles, bins=50, histtype='step', linewidth=2, label=traj_id)
    plt.xlabel('Residues')
    plt.ylabel('Frequency')
    plt.title('Distribution of Angles')
    plt.legend()
    plt.show()

def plot_rg_comparison(rg_data_dict, n_bins=50, bins_range=(1, 4.5), dpi=96):
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    """
    Plot the comparison of radius of gyration (Rg) for multiple ensembles.

    Parameters:
        rg_data_dict (dict): A dictionary where keys are ensemble names and values are 1-D arrays of Rg values.
        n_bins (int): Number of bins for the histogram. Default is 50.
        bins_range (tuple): Range of bins. Default is (1, 4.5).
        dpi (int): Dots per inch, controlling the resolution of the resulting plot. Default is 96.

    Returns:
        None
    """
    h_args = {"histtype": "step", "density": True}
    n_systems = len(rg_data_dict)
    bins = np.linspace(bins_range[0], bins_range[1], n_bins + 1)
    fig, ax = plt.subplots(1, n_systems, figsize=(3 * n_systems, 3), dpi=dpi)
    
    for i, (name_i, rg_i) in enumerate(rg_data_dict.items()):
        ax[i].hist(rg_i, bins=bins, label=name_i, **h_args)
        ax[i].set_title(name_i)
        if i == 0:
            ax[i].set_ylabel("Density")
        ax[i].set_xlabel("Rg [nm]")
        mean_rg = np.mean(rg_i)
        median_rg = np.median(rg_i)

        mean_line = ax[i].axvline(mean_rg, color='k', linestyle='dashed', linewidth=1)
        median_line = ax[i].axvline(median_rg, color='r', linestyle='dashed', linewidth=1)

    
    mean_legend = Line2D([0], [0], color='k', linestyle='dashed', linewidth=1, label='Mean')
    median_legend = Line2D([0], [0], color='r', linestyle='dashed', linewidth=1, label='Median')
    fig.legend(handles=[mean_legend, median_legend], loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_relative_helix_content_multiple_proteins(protein_dssp_data_dict):
    """
    Plot the relative content of H (Helix) in each residue for multiple proteins.

    Parameters:
        protein_dssp_data_dict (dict): A dictionary where keys are protein names and values are 2D arrays with DSSP data.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(next(iter(protein_dssp_data_dict.values())).shape[1])

    for protein_name, dssp_data in protein_dssp_data_dict.items():
        # Count the occurrences of 'H' in each column
        h_counts = np.count_nonzero(dssp_data == 'H', axis=0)
        
        # Calculate the total number of residues for each position
        total_residues = dssp_data.shape[0]
        
        # Calculate the relative content of 'H' for each residue
        relative_h_content = h_counts / total_residues
        
        # Plot the relative content for each protein
        ax.bar(range(len(relative_h_content)), relative_h_content, bottom=bottom, label=protein_name)

        bottom += relative_h_content
    
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Relative Content of H (Helix)')
    ax.set_title('Relative Content of H in Each Residue in the ensembles')
    ax.legend()
    plt.show()

def scatterplot_rg_asphericity(rg_dict, dict):
    """Plot a scatter plot of radius of gyration (rg) vs asphericity."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Creare una griglia di subplot
    
    # Scatter plot di Rg vs Asphericity
    ax1 = axs[0]
    for protein_id in rg_dict.keys():
        rg_values = rg_dict[protein_id]
        asphericity_values = dict[protein_id]
        ax1.scatter(rg_values, asphericity_values, label=protein_id)

    ax1.set_xlabel('Radius of Gyration (Rg)')
    ax1.set_ylabel('{dict}')
    ax1.set_title('Scatter Plot : Rg-{dict}')
    ax1.legend()
    ax1.set_xlim(0,5)
    ax1.set_ylim(0,1)
    
    # Distribuzione di densità per ogni chiave in rg_dict
    ax2 = axs[1]
    sns.set_style("whitegrid")
    for key, values in rg_dict.items():
        sns.kdeplot(values, ax=ax2, label=key)

    ax2.legend()
    ax2.set_xlabel('Rgs')
    ax2.set_ylabel('Density')
    ax2.set_title('Density Distribution of Rgs')

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

def calculate_and_plot_end_to_end_distance(trajs):
    """Calculate the end-to-end distance for each trajectory and plot the distributions."""
    end_to_end_distances = {}

    for traj_id, traj_list in trajs.items():
        for traj in traj_list:
            # Select indices of carbon alpha atoms
            ca_indices = traj.topology.select("protein and name CA")
            # Calculate the Euclidean distance between the end coordinates
            distance = mdtraj.compute_distances(traj, [[ca_indices[0], ca_indices[-1]]])
            end_to_end_distances.setdefault(traj_id, []).extend(distance.flatten())  # Flatten and extend distances
    # Plot distributions
    for traj_id, distances in end_to_end_distances.items():
        sns.kdeplot(distances, label=traj_id)
    plt.xlabel('End-to-End Distance')
    plt.ylabel('Density')
    plt.title('Distribution of End-to-End Distances')
    plt.legend()
    plt.show()
