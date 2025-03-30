# LocatorGraph Test Case - README

This test-case allows you to test LocatorGraph on LLM-like architectures..

---

## 🚀 Step-by-Step Usage

### 1. Record Architecture Graphs (Mutations)

Run the training/mutation recording script to generate the model graphs in `.txt` format.

```
python train_case.py --embedding_dim 128 --num_layers 10 --activation_fn 'gelu'
```

You can run multiple times with different hyperparameters to create different mutations:

```
python train_case.py --embedding_dim 256 --num_layers 4 --activation_fn 'relu'
python train_case.py --embedding_dim 64 --num_layers 6 --activation_fn 'tanh'
...
```

✅ This will generate `.txt` files containing layer-wise architecture graphs for each mutation.

---

### 2. Parse Architecture Graphs to NetworkX Format

Run the provided parsing script to convert recorded `.txt` architecture graphs into **PyTorch Geometric (PyG)** compatible `Data` objects.

```
python parse_to_networkx.py
```

This script will:
- Load all `.txt` files from `Transformer_Graphs_TXT` folder.
- Convert them to **Directed Graphs (DiGraph)** using NetworkX.
- Convert them to **PyTorch Geometric Data format**.
- Save them or prepare them for the next step.

---

### 3. Pass Architecture Graphs to the GNN Model

Once your pre-trained model weights are ready (saved as `.pth` file), you can run inference/testing on your architecture graphs.

```
python pass_to_model.py --weights saved_model.pth
```

✅ This script will:
- Load all `.txt` graphs.
- Convert them to **PyG Data objects**.
- Load the trained GNN model with provided weights.
- **Evaluate** and report **Test Accuracy** on the architecture graphs.

---

**For any questions or improvements, feel free to open an issue or contribute.**