import pandas as pd
import numpy as np
import time
from pymatgen.core import Structure
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from build_graphs import get_c_graph, get_defects_structure, get_visualization
import models
import argparse


def get_prediction(defective_cif_path, pristine_cif_path, actual_value=None, vis=False):
    # Pristine structue
    pristine = Structure.from_file(pristine_cif_path)
    # Defective structure
    defective = Structure.from_file(defective_cif_path)
    # Cloud structure
    cloud = get_defects_structure(defective, pristine)

    if vis:
        get_visualization(pristine, defective, cloud)

    # Build graph
    nodes, edges, edge_features, ids, ratios = get_c_graph(cloud)

    the_data = Data(
        x=torch.tensor(nodes, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        the_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0),
        the_ratios = torch.tensor(ratios, dtype=torch.float).unsqueeze(0)
    )

    attempt = DataLoader([the_data], batch_size=1, shuffle=False)

    # Make prediction
    the_prediction = models.predict(the_model, attempt)[0]

    if actual_value is not None:
        output = f"The predicted band gap: {the_prediction}\t The actual band gap: {actual_value}"
        return output

    else:
        output = f"The predicted band gap: {the_prediction}"
        return output

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
the_model = models.GNNModel().to(device)
model_path = "models/GNN_Model.pth"
the_model.load_state_dict(torch.load(model_path))


def main():
    parser = argparse.ArgumentParser(description="Add the file paths of your pristine and defective structure")

    parser.add_argument("--d", help="File path of your defective structure")
    parser.add_argument("--p", help="File path of your pristine structure")

    arg = parser.parse_args()

    print(get_prediction(arg.d, arg.p))

if __name__ == "__main__":
    main()

