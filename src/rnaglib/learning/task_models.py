import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import RGCNConv, GCNConv, NNConv, GATConv, RGATConv, TransformerConv, global_mean_pool

from rnaglib.utils.misc import tonumpy
#from .edge_rgcn import EdgeRGCNConv


class PygModel(torch.nn.Module):
    @classmethod
    def from_task(
        cls,
        task,
        num_node_features=None,
        num_classes=None,
        graph_level=None,
        multi_label=None,
        **model_args
    ):
        """ Try to create a model based on task metadata.
        Will fail if number of node features is not the default.
        """

        if num_node_features is None:
            num_node_features = task.metadata["num_node_features"]
        if num_classes is None:
            num_classes = task.metadata["num_classes"]
        if graph_level is None:
            graph_level = task.metadata["graph_level"]
        if multi_label is None:
            multi_label = task.metadata["multi_label"]
        return cls(
            num_node_features=num_node_features,
            num_classes=num_classes,
            graph_level=graph_level,
            multi_label=multi_label,
            **model_args
        )

    def __init__(
        self,
        num_node_features,
        num_classes,
        num_unique_edge_attrs=20,
        graph_level=False,
        num_layers=2,
        hidden_channels=128,
        dropout_rate=0.5,
        multi_label=False,
        final_activation="sigmoid",
        layer_type="rgcn",
        num_edge_features=64,
        device=None
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.num_unique_edge_attrs = num_unique_edge_attrs
        self.graph_level = graph_level
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.multi_label = multi_label
        self.layer_type = layer_type
        self.num_edge_features = num_edge_features

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation == "softmax":
            self.final_activation = torch.nn.Softmax(dim=1)
        else:
            self.final_activation = torch.nn.Identity()

        # Input layer

        self.input_non_linear_layer = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
        )

        for i in range(self.num_layers):                
            if self.layer_type in ["gcn", "GCN"]:
                self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
            elif self.layer_type in ["gat", "GAT", "edge_gat", "edge_GAT"]:
                self.convs.append(GATConv(self.hidden_channels, self.hidden_channels))
            elif self.layer_type in ["edge_gcn","edge_GCN"]:
                self.nn = torch.nn.Sequential(torch.nn.Linear(self.num_unique_edge_attrs, 32), torch.nn.ReLU(), torch.nn.Linear(32, self.hidden_channels**2))
                self.convs.append(NNConv(self.hidden_channels, self.hidden_channels, self.nn))
            # elif self.layer_type in ["edge_rgcn","edge_RGCN"]:
            #     self.convs.append(EdgeRGCNConv(self.hidden_channels, self.hidden_channels, self.num_unique_edge_attrs, self.num_edge_features))
            elif self.layer_type in ["transformer"]:
                self.convs.append(TransformerConv(self.hidden_channels, self.hidden_channels))
            elif self.layer_type in ["rgat", "RGAT", "edge_rgat", "edge_RGAT"]:
                self.convs.append(RGATConv(self.hidden_channels, self.hidden_channels, self.num_unique_edge_attrs, edge_dim=self.num_edge_features))
            else:
                self.convs.append(RGCNConv(self.hidden_channels, self.hidden_channels, self.num_unique_edge_attrs))
            self.bns.append(BatchNorm1d(self.hidden_channels))
            self.dropouts.append(Dropout(self.dropout_rate))

        # Initialize training components
        # Output layer
        if self.multi_label:
            self.final_linear = torch.nn.Linear(self.hidden_channels, self.num_classes)
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.final_activation = torch.nn.Identity()  # Use Identity for multi-label
        elif num_classes == 2:
            self.final_linear = torch.nn.Linear(self.hidden_channels, 1)
            # Weight will be set in train_model based on actual class distribution
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.final_activation = torch.nn.Identity()  # Use Identity for binary
        else:
            self.final_linear = torch.nn.Linear(self.hidden_channels, self.num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()
            if final_activation == "sigmoid":
                self.final_activation = torch.nn.Sigmoid()
            elif final_activation == "softmax":
                self.final_activation = torch.nn.Softmax(dim=1)
            else:
                self.final_activation = torch.nn.Identity()

        self.optimizer = None
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.configure_training()

    def forward(self, data, return_mask=False):
        x, edge_index, edge_attrs, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_batch = batch[edge_index[0]]

        if hasattr(data,"edge_feats"):
            # Per-edge: does this edge have a NaN?
            nan_mask_edges = torch.isnan(data.edge_feats).any(dim=1).float()  # [num_edges]

            # Per-graph: does any edge in this graph have a NaN? (max reduce: 1.0 if any NaN)
            num_graphs = batch.max().item() + 1
            graph_has_nan = torch.zeros(num_graphs, device=batch.device).scatter_reduce(
                0, edge_batch, nan_mask_edges, reduce="amax"
            ).bool()

            # Propagate back to edges
            clean_edge_mask = ~graph_has_nan[edge_batch]  # [num_edges]

            # Zero out edges belonging to graphs with NaNs
            edge_feats = data.edge_feats.clone()
            edge_feats[~clean_edge_mask] = 0.0

        x = self.input_non_linear_layer(x)

        for i in range(self.num_layers):
            
            if self.layer_type in ["gcn", "GCN", "gat", "GAT"]:
                x = self.convs[i](x, edge_index)
            elif self.layer_type in ["edge_rgcn", "edge_RGCN", "edge_rgat", "edge_RGAT"]:
                x = self.convs[i](x, edge_index, edge_attrs, edge_feats)
            elif self.layer_type in ["edge_gat", "edge_GAT", "edge_gcn", "edge_GCN"]:
                x = self.convs[i](x, edge_index, edge_feats)
            else:
                x = self.convs[i](x, edge_index, edge_attrs)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)

        if self.graph_level:
            x = global_mean_pool(x, batch)
        x = self.final_linear(x)
        x = self.final_activation(x)
        if return_mask:
            # For graph-level: check if batch has edge_feats attribute
            # For node-level: create mask based on whether data has edge_feats
            if self.graph_level:
                # Assuming you add a flag to the batch data
                mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
                if hasattr(data, 'has_edge_feats_mask'):
                    mask = data.has_edge_feats_mask
            else:
                # For node-level, all nodes in a graph share the same edge features status
                #mask = ~torch.isnan(data.edge_feats).any(dim=1)
                mask =  ~graph_has_nan[batch]
            return x, mask
        return x

    def configure_training(self, learning_rate=0.001):
        """Configure training settings."""
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)  # Move criterion to device for all cases
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def compute_loss(self, out, target, weighted=False, metadata=None, mask=None):
        # Apply mask if provided
        if mask is not None:
            out = out[mask]
            target = target[mask]
            # If no valid samples remain, return zero loss
            if out.size(0) == 0:
                return torch.tensor(0.0, device=out.device, requires_grad=True)
            
        # If just two classes, flatten outputs since BCE behavior expects equal dimensions and CE (N,k):(N)
        # Otherwise CE expects long as outputs
        if not self.multi_label:
            if self.num_classes == 2:
                out = out.flatten()
            else:
                target = target.long()
        loss = self.criterion(out, target)
        if weighted:
            batch_class_distribution = {i: (target == i).sum().item() for i in range(metadata['num_classes'])}
            batch_weight = 0
            num_classes = metadata['num_classes']
            for i in range(num_classes):
                batch_weight += 1 / metadata["class_distribution"][str(i)] * batch_class_distribution[i] / len(target)
            batch_weight = batch_weight * metadata["dataset_size"] / num_classes
            loss = loss * batch_weight
        return loss

    def train_model(self, task, epochs=500):
        if self.optimizer is None:
            self.configure_training()

        # Set class weights for binary classification based on actual distribution
        if self.num_classes == 2:
            neg_count = float(task.metadata["class_distribution"]["0"])
            pos_count = float(task.metadata["class_distribution"]["1"])
            pos_weight = torch.tensor(np.sqrt(neg_count / pos_count)).to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for epoch in range(epochs):
            # Training phase
            self.train()
            epoch_loss = 0
            num_batches = 0
            for batch in task.train_dataloader:
                graph = batch["graph"].to(self.device)
                self.optimizer.zero_grad()
                if "edge_feats" in graph:
                    out, mask = self(graph, return_mask=True)
                    loss = self.compute_loss(out, graph.y, mask=mask)
                else:
                    out = self(graph)
                    loss = self.compute_loss(out, graph.y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            # Validation phase
            if epoch % 10 == 0:
                val_metrics = self.evaluate(task, split="val")
                print(
                    f"Epoch {epoch}: train_loss = {epoch_loss / num_batches:.4f}, val_loss = {val_metrics['loss']:.4f}",
                )

    def inference(self, loader) -> tuple:
        """Evaluate model performance on a dataset.

        Args:
            loader: Data loader to use

        Returns:
            3 list containing predictions, probs, targets if residue level,
        else 3 np arrays
        """
        self.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        total_loss = 0
        num_valid_samples = 0
        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"]
                graph = graph.to(self.device)
                labels = graph.y
                if "edge_feats" in graph:
                    out, mask = self(graph, return_mask=True)
                    loss = self.compute_loss(out, labels, mask=mask)
                else:
                    out = self(graph)
                    loss = self.compute_loss(out, labels)
                total_loss += loss.item()

                # For binary/multilabel, threshold the logits at 0 (equivalent to prob > 0.5 after sigmoid)
                preds = (out > 0).float() if (self.multi_label or self.num_classes == 2) else out.argmax(dim=1)
                probs = out

                probs = tonumpy(probs)
                preds = tonumpy(preds)
                labels = tonumpy(labels)
                mask_np = tonumpy(mask)

                # split predictions per RNA if residue level
                if not self.graph_level:
                    cumulative_sizes = tuple(tonumpy(graph.ptr))
                    probs = [
                        probs[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                    preds = [
                        preds[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                    labels = [
                        labels[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                    mask_np = [mask_np[start:end] for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)]
            
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        if self.graph_level:
            all_probs = np.stack(all_probs)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
        mean_loss = total_loss / len(loader)
        return mean_loss, all_preds, all_probs, all_labels

    def get_dataloader(self, task, split="test"):
        if split == "test":
            dataloader = task.test_dataloader
        elif split == "val":
            dataloader = task.val_dataloader
        else:
            dataloader = task.train_dataloader
        return dataloader

    def evaluate(self, task, split="test"):
        dataloader = self.get_dataloader(task=task, split=split)
        mean_loss, all_preds, all_probs, all_labels = self.inference(loader=dataloader)
        metrics = task.compute_metrics(all_preds, all_probs, all_labels)
        metrics["loss"] = mean_loss
        return metrics
