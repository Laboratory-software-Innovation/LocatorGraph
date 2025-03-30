import os
import ast
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader

def load_txt_graphs_as_pyg(folder_path, labels=None):
    pyg_graphs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            G = nx.DiGraph()
            prev_node = None

            with open(path, "r") as f:
                for line in f:
                    if line.strip().startswith("("):
                        layer, attrs = ast.literal_eval(line.strip())
                        node_id = f"{layer}_{len(G.nodes)}"
                        G.add_node(node_id, layer=layer)
                        if prev_node:
                            G.add_edge(prev_node, node_id)
                        prev_node = node_id

            pyg_graph = from_networkx(G)

            num_nodes = pyg_graph.num_nodes
            pyg_graph.x = torch.eye(num_nodes)

            pyg_graph.y = torch.tensor([label for label in labels], dtype=torch.long) if labels else torch.zeros(num_nodes, dtype=torch.long)

            pyg_graphs.append(pyg_graph)

    return pyg_graphs

def load_pretrained_model(weight_path, num_node_features, ModelClass=None):
    model = ModelClass(num_node_features)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    for batch in test_loader:
        out = model(batch)
        pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())
        total += batch.num_graphs
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")

graph_folder = "Transformer_Graphs_TXT"
weight_path = "saved_model.pth"  

graphs = load_txt_graphs_as_pyg(graph_folder)
test_loader = DataLoader(graphs, batch_size=4)

num_node_features = graphs[0].num_node_features
model = load_pretrained_model(weight_path, num_node_features)

evaluate_model(model, test_loader)
