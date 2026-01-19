import os

import pickle
import torch
import networkx as nx
import numpy as np

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.algorithms import fix_buggy_edges

from .graph import GraphRepresentation

class DistographRepresentation(GraphRepresentation):

    def __init__(
        self,
        distograms_path,
        distogram_edges=True,
        distogram_edge_features=False,
        B=10,
        tau=1e-4,
        distogram_files_prefix="distogram_",
        distogram_files_suffix="_model_0.pkl",
        framework="nx",
        clean_edges=True,
        edge_map=GRAPH_KEYS["edge_map"][TOOL],
        etype_key="LW",
        **kwargs,
    ):
        self.distograms_path = distograms_path
        self.distogram_edges = distogram_edges
        self.distogram_edge_features = distogram_edge_features
        self.B = B
        self.tau = tau
        self.distogram_files_prefix = distogram_files_prefix
        self.distogram_files_suffix = distogram_files_suffix
        super().__init__(framework, clean_edges, edge_map, etype_key, **kwargs)
        pass

    def __call__(self, rna_graph, features_dict):

        if self.clean_edges:

            base_graph = fix_buggy_edges(graph=rna_graph, label=self.etype_key, edge_map=self.edge_map)

        else:

            base_graph = rna_graph

        if self.framework == "nx":

            raise ValueError("Distographs only supported with pyg framework, not networkx")
        
        if self.framework == "dgl":

            raise ValueError("Distographs only supported with pyg framework, not dgl")
        
        if self.framework == "pyg":

            pyg_graph = super().to_pyg(base_graph, features_dict)

            with open(os.path.join(self.distograms_path,f"{self.distogram_files_prefix}{rna_graph.name}{self.distogram_files_suffix}"),"rb") as f:

                distogram_dict = pickle.load(f)
                distogram = distogram_dict["distogram"]["softmax"]

            if self.distogram_edges:

                proba_matrix = distogram[:,:,:self.B].sum(axis=2)
                proba_matrix.fill_diagonal_(float(0))
                new_edges = torch.nonzero(proba_matrix > self.tau, as_tuple=False)
                new_edges = new_edges.t()

                new_edge_attr = torch.full((new_edges.size(1),), max(self.edge_map.values())+1, dtype=torch.long)
                pyg_graph.edge_index = torch.cat([pyg_graph.edge_index, new_edges], dim=1)
                pyg_graph.edge_attr = torch.cat([pyg_graph.edge_attr, new_edge_attr], dim=0)

            if self.distogram_edge_features:
                
                row, col = pyg_graph.edge_index
                edge_distances = torch.from_numpy(distogram[row,col])
                pyg_graph.edge_attr = edge_distances
                #pyg_graph.edge_attr = torch.cat([pyg_graph.edge_attr.unsqueeze(1), edge_distances], dim=1)

            return pyg_graph

    def batch(self, samples):
        """
        Batch a list of graph samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        if self.framework == "nx":
            raise ValueError("Distographs only supported with pyg framework, not networkx")
        if self.framework == "dgl":
            raise ValueError("Distographs only supported with pyg framework, not dgl")
        if self.framework == "pyg":
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(samples)
            # sometimes batching changes dtype from int to float32?
            batch.edge_index = batch.edge_index.to(torch.int64)
            if self.distogram_edge_features:
                batch.edge_attr = batch.edge_attr.to(torch.float32)
            else:
                batch.edge_attr = batch.edge_attr.to(torch.int64)
            return batch


        