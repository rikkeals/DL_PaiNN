"""
Helper functions for the PaiNN model

Functions: 
Local Edges - Defines atom's neighbors within a molecule
RDF - Computes the radial distribution function (RDF) for distance between atompairs

"""

import torch

def local_edges(atom_positions, 
                graph_indexes,
                cutoff_dist, 
                device):
    """
    Computes the local edges for a set of atom positions and graph indexes.

    Args:
        atom_positions (torch.Tensor): Tensor of shape (N, 3) containing the positions of N atoms.
        graph_indexes (torch.Tensor): Tensor of shape (M, 2) containing the indexes of M edges.
        cutoff_dist (float): The cutoff distance for defining local edges.
        device (str): The device to perform computations on ('cpu' or 'cuda').

    Returns:
        edge_indexes (torch.Tensor): Tensor of shape (K, 2) containing the indexes of K edges that are valid for message passing.
        edge_distances (torch.Tensor): Tensor of shape (K,) containing the distances of the K edges.
        edge_directions (torch.Tensor): Tensor of shape (K, 3) containing the direction vectors of the K edges.
    """
    
    # Number of atoms
    num_atoms = graph_indexes.shape[0]

    # Pairwise comparisson between all atom to find which ones are neighbors
    pos_i = atom_positions.unsqueeze(0).repeat(num_atoms, 1, 1)
    pos_j = atom_positions.unsqueeze(1).repeat(1, num_atoms, 1)

    # Compute the relative positions and distances between all atom pairs
    rel_pos_ij = (pos_j - pos_i).to(device)
    dist_ij = torch.norm(rel_pos_ij, dim=2)

    # Masks neeeded for atom pairs
    # Within the cutoff distance
    cutoff_mask = dist_ij <= cutoff_dist
    # Not self-interaction
    self_interaction_mask = torch.arange(num_atoms).unsqueeze(0) != torch.arange(num_atoms).unsqueeze(1)
    # In same molecule
    same_molecule_mask = graph_indexes.unsqueeze(0) == graph_indexes.unsqueeze(1)

    #Make sure they are on the same device
    cutoff_mask = cutoff_mask.to(device)
    self_interaction_mask = self_interaction_mask.to(device)
    same_molecule_mask = same_molecule_mask.to(device)

    # Combine masks to get valid edges for message passing
    valid_edges_mask = cutoff_mask & self_interaction_mask & same_molecule_mask

    # Compute the edges needed for message passing
    edge_indexes = valid_edges_mask.nonzero(as_tuple=False).t()
    edge_distances = dist_ij[valid_edges_mask]
    edge_directions = rel_pos_ij[valid_edges_mask]

    # Return the edges and their properties
    return edge_indexes, edge_distances, edge_directions


def RDF(edge_distances, 
        num_rbf_features,
        cutoff_dist,
        device):
    """
    Computes the radial distribution function (RDF) for a set of edge distances, and 
    thereby expands distance into a learnable basis function. 

    Args: 
        edge_distances (torch.Tensor): Tensor of shape (K,) containing the distances of K edges.
        num_rbf_features (int): The number of radial basis function features to compute.
        cutoff_dist (float): The cutoff distance for defining the RDF.
        device (str): The device to perform computations on ('cpu' or 'cuda').

    Returns:
        edge_rdf (torch.Tensor): Tensor of shape (num_rbf_features,) containing the RDF values for the edge distances.

    """

    # Number of local edges
    num_edges = edge_distances.size(0)

    # Create a tensor of evenly spaced RBF frequencies from 1 to 20
    n_values = torch.linspace(1, 20, num_rbf_features, device=device)

    # Expand n_values to match number of edges for element-wise RBF computation
    n_values_expanded = n_values.unsqueeze(0).expand(num_edges, num_rbf_features)

    # Expand edge distances to match n_values for broadcasting
    edge_distances_expanded = edge_distances.unsqueeze(1).expand(num_edges, num_rbf_features)

    # Compute the sinusiodal RDF values for each pair of edges
    edge_rbf = torch.sin(n_values_expanded * torch.pi * edge_distances_expanded / cutoff_dist) / (edge_distances_expanded)

    return edge_rbf