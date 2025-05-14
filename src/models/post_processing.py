import torch
import torch.nn as nn


class AtomwisePostProcessing(nn.Module):
    """
    Post-processing for (QM9) properties that are predicted as sums of atomic
    contributions. Can handle cases where atomic references are not available.
    """
    def __init__(
        self,
        num_outputs: int,
        mean: torch.FloatTensor,
        std: torch.FloatTensor,
        atom_refs: torch.FloatTensor = None,  # <- allow it to be None
    ) -> None:
        """
        Args:
            num_outputs: Number of model outputs. In most cases 1.
            mean: Mean value to shift atomwise contributions by.
            std: Standard deviation to scale atomwise contributions by.
            atom_refs: (Optional) Atomic reference values. If None, skip this correction.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.register_buffer('scale', std)
        self.register_buffer('shift', mean)

        if atom_refs is not None:
            self.atom_refs = nn.Embedding.from_pretrained(atom_refs, freeze=True)
        else:
            self.atom_refs = None

    def forward(
        self,
        atomic_contributions: torch.FloatTensor,
        atoms: torch.LongTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Atomwise post-processing operations and atomic sum.

        Args:
            atomic_contributions: [num_nodes, num_outputs] each atom's contribution.
            atoms: [num_nodes] atom type of each node.
            graph_indexes: [num_nodes] graph index each node belongs to.

        Returns:
            [num_graphs, num_outputs] predictions for each graph (molecule).
        """
        num_graphs = torch.unique(graph_indexes).shape[0]

        atomic_contributions = atomic_contributions * self.scale + self.shift

        if self.atom_refs is not None:
            atomic_contributions = atomic_contributions + self.atom_refs(atoms)

        # Sum atomic contributions into per-graph output
        output_per_graph = torch.zeros(
            (num_graphs, self.num_outputs),
            device=atomic_contributions.device,
        )
        output_per_graph.index_add_(
            dim=0,
            index=graph_indexes,
            source=atomic_contributions,
        )

        return output_per_graph
