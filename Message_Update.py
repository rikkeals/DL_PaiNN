"""
Message and Update functions for Painn Model

"""

import torch
import torch.nn as nn

class Message(nn.Module): 
    """
    Message function for Painn Model

    Args: 
    Self (nn.Module): Inherits from nn.Module
    num_features (int): Number of features in the input data
    num_rbf_features (int): Number of radial basis function features
    device (str): Device to run the model on (e.g., 'cuda' or 'cpu')

    Returns:
    dsf (torch.Tensor): Output tensor after applying the message function
    dvf (torch.Tensor): Vector of distances between nodes

    """
    def __init__(self,
                 num_features: int,
                 num_rbf_features: int,
                 device: str):
        super().__init__()

        self.num_features = num_features
        self.num_rbf_features = num_rbf_features
        self.device = device

        # Linear layers for scalar features (sf) for each atom
        # and expanding to 3 times the number of features
        self.linear_sf = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )

        # Linear layer for radial basis function (rbf) features
        self.linear_rbf = nn.Linear(num_rbf_features, num_features * 3)
        # Now both sf and rbf have the same shape: num_edges x num_features * 3, 
        # so they can be combined in message computations

    def CosineCutoff(self,
                     edge_distance, 
                     cutoff_dist):
        """
        Cosine cutoff function to apply a cutoff distance to the edges.
        Args:
            edge_distance (torch.Tensor): Distances between the edges.
            cutoff_dist (float): Cutoff distance for the message passing.

        Returns:
            CosCut (torch.Tensor): Cutoff function values.
        """
        # Cosine cutoff function
        CosCut = 0.5 * (1 + torch.cos(torch.pi * edge_distance / cutoff_dist))

        return CosCut


    def forward(self,
                sf, 
                vf,
                edge_indexes,
                edge_vector,
                edge_distance,
                edge_rbf,
                cutoff_dist):
        """
        Forward pass of the Message function. Computes the message for each atom based on its neighbors.
        Args:
            sf (torch.Tensor): Scalar features of the atoms.
            vf (torch.Tensor): Vector features of the atoms.
            edge_indexes (torch.Tensor): Edge indexes for the message passing.
            edge_vector (torch.Tensor): Vector features of the edges.
            edge_distance (torch.Tensor): Distances between the edges.
            edge_rbf (torch.Tensor): Radial basis function features of the edges.
            cutoff_dist (float): Cutoff distance for the message passing.

        Returns:
            dsf (torch.Tensor): Scalar features after message passing.
            dvf (torch.Tensor): Vector features after message passing.
        
        """

        # Number of atoms in the batch
        num_atoms = sf.size(0)

        # Make empty tensors for the outputs, dsf and dvf
        dsf = torch.zeros(num_atoms, self.num_features).to(self.device)
        dvf = torch.zeros(num_atoms, 3, self.num_features).to(self.device)

        # Gather the scalar features (sf) and vector features (vf) of the neighbors
        # based on the edge indexes so there is one neighbor for each edge
        Neighbors_sf = sf[edge_indexes[1]]
        Neighbors_vf = vf[edge_indexes[1]]

        # Applying the linear layers to the neighbors' scalar features
        phi = self.linear_sf(Neighbors_sf)
        # Linear combination of the radial basis functions
        edge_rbf_linear = self.linear_rbf(edge_rbf)

        # Define the Cosine cutoff function
        coscut = self.CosineCutoff(edge_distance, cutoff_dist)

        # Scale the features with the cutoff function
        W = edge_rbf_linear * coscut[..., None]

        final_message = W * phi

        # Split the final message into three parts: Wsf, Wvf_vf and Wvf_sf
        Wsf, Wvf_vf, Wvf_sf = torch.split(final_message, self.num_features, dim=-1)

        # Aggregate the contributions from neighboring atoms (scalar feature)
        # to update the scalar features of each atom
        dsf = dsf.index_add_(dim=0, index=edge_indexes[0], source=Wsf, alpha=1.0)

        # Normalize edge vectors to unit length seperates direction from distance
        # to keep the direction of the vector features
        edge_vector = edge_vector / edge_distance[..., None]

        # Total edge-wise directional update pr. feature
        # computed by mixing the vector features of the neighbors and the edge vectors
        # using the weights Wvf_vf and Wvf_sf
        dvec = Wvf_vf.unsqueeze(1) * Neighbors_vf + edge_vector.unsqueeze(2) * Wvf_sf.unsqueeze(1)

        # Aggregate the contributions from neighboring atoms (vector feature)
        # to update the vector features of each atom
        dvf = dvf.index_add_(dim=0, index=edge_indexes[0], source=dvec, alpha=1.0)

        return dsf, dvf
        

class Update(nn.Module):
    """
    Update function for Painn Model

    Args: 
    Self (nn.Module): Inherits from nn.Module
    num_features (int): Number of features in the input data
    device (str): Device to run the model on (e.g., 'cuda' or 'cpu')

    Returns:
    dsf (torch.Tensor): Output tensor after applying the update function
    dvf (torch.Tensor): Vector of distances between nodes

    """
    def __init__(self,
                 num_features: int,
                 device: str):
        super().__init__()

        self.num_features = num_features
        self.device = device

        # Linear layers for vector features (vf) for each atom
        # and expanding to two times the number of features
        self.linear_vf = nn.Sequential(
            nn.Linear(num_features, num_features*2, bias = False)
        )

        # Linear layers for scalar and vector features (sf and vf) for each atom
        # and expanding to 3 times the number of features
        self.linear_sf_vf = nn.Sequential(
            nn.Linear(2*num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features*3)
        )

    def forward(self, 
                dsf, 
                dvf, 
                sf, 
                vf):
        """
        Forward pass of the Update function. Computes the update for each atom based on its features.
        Args:
            self (nn.Module): Instance of the Update class.
            sf (torch.Tensor): Scalar features of the atoms.
            vf (torch.Tensor): Vector features of the atoms.

        Returns:
            dsf (torch.Tensor): Updated scalar features after message passing.
            dvf (torch.Tensor): Updated vector features after message passing.
        
        """
        # Linear combinations of the vector features
        vf = self.linear_vf(vf)

        # Split the vector features into two parts: vf_U and vf_V
        vf_U, vf_V = torch.split(vf, self.num_features, dim=-1)

        # Compute dot product of vector V and vector U across spatial dimensions
        dot_vf = (vf_U * vf_V).sum(dim=1)

        # Compute Euclidean norm of each vector in vf_V, across spatial dimensions
        # Epsilon = 1e-8 to avoid division by zero
        norm_vf = torch.sqrt(torch.sum(vf_V**2, dim=1)+ 1e-8)

        # Applying the linear layers to the scalar and vector features
        vec_W = self.linear_sf_vf(torch.cat([sf, norm_vf], dim=-1))

        # Split the final message into three parts: Wsf, Wvf_vf and Wvf_sf
        Wsf, Wvf_vf, Wvf_sf = torch.split(vec_W, self.num_features, dim=-1)

        # Compute the final change in scalar feature 
        dsf = Wsf + dot_vf * Wvf_vf

        # Compute the final change in vector feature
        dvf = Wvf_vf.unsqueeze(1) * vf_U

        return dsf, dvf
            


    