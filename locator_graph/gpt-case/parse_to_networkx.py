import os
import ast
import networkx as nx
import matplotlib.pyplot as plt

graph_folder = "Transformer_Graphs_TXT"

def load_graph_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    graph = nx.DiGraph()
    prev_node = None
    for line in lines:
        if line.strip().startswith('('):
            layer_info = ast.literal_eval(line.strip())
            layer_name = layer_info[0]
            attributes = layer_info[1]
            node_id = f"{layer_name}_{len(graph.nodes)}"
            graph.add_node(node_id, **attributes, layer=layer_name)
            if prev_node:
                graph.add_edge(prev_node, node_id)
            prev_node = node_id
    return graph

all_graphs = {}
folder_path = "./Transformer_Graphs_TXT"  # Change if path is different
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        graph_name = file.replace(".txt", "")
        file_path = os.path.join(folder_path, file)
        all_graphs[graph_name] = load_graph_from_txt(file_path)

print(f"Loaded {len(all_graphs)} graphs.")

def plot_graph(graph, title="Graph"):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    labels = {node: data["layer"] for node, data in graph.nodes(data=True)}
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=1000, node_color="lightblue", arrows=True)
    plt.title(title)
    plt.show()

first_graph_name = list(all_graphs.keys())[0]
plot_graph(all_graphs[first_graph_name], title=first_graph_name)
