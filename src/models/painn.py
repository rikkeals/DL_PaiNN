"""
The Painn Model Implementation.
Copyright (c) 2023, The University of Cambridge and the authors of the 
Polarizable Atom Interaction Neural Network (PaiNN) paper.
All rights reserved.
"""



import torch
import torch.nn as nn
from helper_functions import local_edges, RDF
from Message_Update import Message, Update


class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
        num_output: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
            num_output: Number of outputs for the model. 

            device: Device to run the model on either 'cuda' or 'cpu'

        """

        # Initialize the PaiNN model with the given parameters.
        super().__init__()

        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist
        self.device = device

        # Translate the atom types to a one-hot encoding, so its not letters but numbers
        self.atom_embedding = nn.Embedding(num_unique_atoms, num_features)

        # Initialize the message and update blocks for the model.
        self.message = nn.ModuleList()
        self.update = nn.ModuleList()

        # Loop through the number of message passing layers and create the message and update blocks.
        # Number of layers shows how many neighbors we want to consider in the message passing.
        for i in range(num_message_passing_layers):
            self.message.append(
                Message(
                    num_features,
                    num_rbf_features,
                    device,
                )
            )
            self.update.append(
                Update(
                    num_features,
                    device,
                )
            )
        
        # Initialize the output layer for the model.
        # The output layer is a linear layer that takes the final node features and outputs the predicted property.
        self.output = nn.Sequential(
            nn.Linear(num_features, num_features//2),
            nn.SiLU(),
            nn.Linear(num_features//2, num_output),
        )
     


    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        
        # Learn the atom embeddings for the input atoms.
        sf = self.atom_embedding(atoms).to(self.device)
        vf = torch.zeros(sf.size(0),3,sf.size(1)).to(self.device)

        ##### Local neigborhood #####
        # Get the local edges for the input atoms using the helper function.
        edge_indexes, edge_distances, edge_directions = local_edges(
            atom_positions,
            graph_indexes,
            self.cutoff_dist,
            self.device
        )

        ###### Radial Basis Function (RBF) #####
        # Compute the radial distribution function (RDF) for the input atoms using the helper function.
        edge_rbf = RDF(
            edge_distances,
            self.num_rbf_features,
            self.cutoff_dist,
            self.device
        )

        # Move the tensors to the appropriate device
        edge_indexes = edge_indexes.to(self.device)
        edge_distances = edge_distances.to(self.device)
        edge_directions = edge_directions.to(self.device)
        edge_rbf = edge_rbf.to(self.device)

        ##### Message and Update #####
        # Loop through the number of message passing layers and perform message passing and update steps.
        for i in range(self.num_message_passing_layers):
            # Message passing step
            dsf, dvf = self.message[i](
                sf,
                vf,
                edge_indexes,
                edge_directions,
                edge_distances,
                edge_rbf,
                self.cutoff_dist
            )

            sf = sf + dsf
            vf = vf + dvf

            # Update step
            sf, vf = self.update[i](
                dsf,
                dvf,
                sf,
                vf
            )

            sf = sf + dsf
            vf = vf + dvf

        ##### Output #####
        # Compute the output for the model using the final node features.
        atomic_contributions = self.output(sf)

        return atomic_contributions
