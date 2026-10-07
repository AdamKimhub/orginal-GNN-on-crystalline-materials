import numpy as np
import pandas as pd
from pymatgen.core import Structure, PeriodicSite, DummySpecie, Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import MinimumDistanceNN
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor


# Visualize the structures
def get_visualization(pristine, defective, cloud):
    fig, num = plt.subplots(1,3, figsize=(15,5))
    # Visualize the pristine structure
    new_pristine = AseAtomsAdaptor.get_atoms(pristine)
    plot_atoms(new_pristine, num[0])
    num[0].set_title(f"Pristine {pristine.formula}")

    # Visualize the defective structure
    new_defective = AseAtomsAdaptor.get_atoms(defective)
    plot_atoms(new_defective, num[1])
    num[1].set_title(f"Defective Structure: {defective.formula}")

    # Visualize the structure without defects
    new_cloud = AseAtomsAdaptor.get_atoms(cloud)
    plot_atoms(new_cloud, num[2])
    num[2].set_title(f"Cloud Structure(Defects only structure): {cloud.formula}")

def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}

"""
# Get formation energy of all elements in the periodiic table
import config
from mp_api.client import MPRester
from pymatgen.core.periodic_table import Element

all_elements = [str(el) for el in Element]

API_KEY = config.API_KEY # Replace with your Materials Project API key

def get_formation(element, API_KEY):
    with MPRester(API_KEY) as mpr:
        results = mpr.materials.summary.search(
            elements=[element],
            num_elements=1,
            fields= ["energy_per_atom"]
        )
        forms_list = [result.energy_per_atom for result in results]
        avg_formation_energy = np.mean(forms_list)

    return avg_formation_energy

formation_energies = {element: get_formation(element, API_KEY) for element in all_elements}
print(formation_energies)
"""
formation_energies = {
    "H":-2.835951846430921, "He":-0.31196323625, "Li":-2.3461654534259258, "Be":-3.6906741816666666,
    "B":-7.002827127491851, "C":-9.316731934117035, "N":-7.233842272742187, "O":-4.676438329750001,
    "F":-2.9296551798214288, "Ne":-1.91282416, "Na":-3.4536177056717374, "Mg":-4.09163935124074,
    "Al":-6.63584847076, "Si":-8.426176753356291, "P":-8.31567715169643, "S":-7.827358545769599,
    "Cl":-6.128105756041667, "Ar":-4.856792372499999, "K":-5.951273052253347, "Ca":-6.8788465471875,
    "Sc":-11.237957669242425, "Ti":-12.802568009213335, "V":-13.922600690000001, "Cr":-14.77669376875,
    "Mn":-14.348056304012763, "Fe":-8.25899893927, "Co":-13.16277911966809, "Ni":-11.659201161666667,
    "Cu":-10.54583238875, "Zn":-8.8732424244375, "Ga":-11.365511257524622, "Ge":-13.659562455970299,
    "As":-14.343474869861112, "Se":-14.136485254729166, "Br":-2.802858366809896, "Kr":-12.577137455166667,
    "Rb":-3.522684412102679, "Sr":-14.872080059242423, "Y":-20.267552732083335, "Zr":-22.357979443055555,
    "Nb":-24.592838769404764, "Mo":-25.65156121359375, "Tc":-25.938798287500003, "Ru":-25.359055447499998,
    "Rh":-23.891668894, "Pd":-22.939867830625, "Ag":-21.344974299333337, "Cd":-20.075181660000002,
    "In":-22.569768360619864, "Sn":-24.623966149094205, "Sb":-25.500892143392857, "Te":-25.35888922212963,
    "I":-5.240995872916667, "Xe":-2.681022067222222, "Cs":-25.08106868751572, "Ba":-1.714427234090909,
    "La":-29.703585855, "Ce":-30.765536582916667, "Pr":-29.360978606000003, "Nd":-29.38402319895833,
    "Pm":-29.532344363055557, "Sm":-29.722466492916666, "Eu":-38.489393041388894, "Gd":-43.33816982966667,
    "Tb":-31.19427302633333, "Dy":-31.949257627833333, "Ho":-32.837879401, "Er":-33.877142851833334,
    "Tm":-35.078865518166666, "Yb":-36.496195276250006, "Lu":-38.04532439777778, "Hf":-45.06808062583333,
    "Ta":-47.144759839058324, "W":-50.771654835674994, "Re":-52.330339372333334, "Os":-51.948078458750004,
    "Ir":-51.251150415, "Pt":-51.429006055, "Au":-50.54936647375, "Hg":-49.346659606686515, "Tl":-53.16799786356322, 
    "Pb":-56.20973389190476, "Bi":-58.259812768529414, "Ac":-68.625217766875, "Th":-73.399498815, 
    "Pa":-76.84324669166666,    "U":-79.60085220729613, "Np":-82.84298793625, "Pu":-86.04870866293301, 
    "Am":0, "Cm":0, "Bk":0, "Cf":0,"Es":0, "Fm":0, "Md":0, "No":0, "Lr":0, "Rf":0, 
    "Db":0, "Sg":0, "Bh":0, "Hs":0, "Mt":0, "Ds":0, "Rg":0, "Cn":0, "Nh":0, "Fl":0,
    "Mc":0, "Lv":0, "Ts":0, "Og":0, "Po": 0, "At":0, "Rn":0, "Fr":0, "Ra":0
}

def fe_site(original, new):
    if new == 0: # For vcancy
        fe_defect = formation_energies[original] * -1

    else: # For substitution
        form_original = formation_energies[original]
        form_new = formation_energies[new]
        fe_defect = (form_original * -1) + form_new

    return fe_defect


def get_defects_structure(defective_struct, reference_struct):
    mindnn = MinimumDistanceNN()
    # struct to dict
    defective_dict = struct_to_dict(defective_struct)
    reference_dict = struct_to_dict(reference_struct)

    # Get lattice of defective structure
    structure_lattice = defective_struct.lattice

    # List to add all defect sites
    defects_list = []

    # Dictionary to hold properties of each defect site
    defects_properties = {} 

    for ref_coord, ref_site in reference_dict.items():
        # Use the reference coordinates to get the defective site
        def_site = defective_dict.get(ref_coord)

        if def_site:  # The site is found in both the reference structure and the defective structure
            # But are the species the same?
            if ref_site.specie != def_site.specie:  # Substitution
                # Add site to defects list
                defects_list.append(def_site)

                # Get atomic number change and defect type
                add_property = {"original_an":ref_site.specie.Z,
                                "new_an": def_site.specie.Z,
                                "an_change": def_site.specie.Z - ref_site.specie.Z,
                                "vacancy_defect": 0.0,
                                "substitution_defect": 1.0,
                                "bonds_broken": 0.0,
                                "site_fe": fe_site(ref_site.species_string,def_site.species_string)}
                defects_properties[def_site] = add_property

        else: # the site from ref_structure is not found in defective structure
            # This means that the site is a vacancy site
            # Add site to defective structure
            vacant_site = PeriodicSite(
                species= DummySpecie(),
                coords= ref_coord,
                coords_are_cartesian= False, 
                lattice= structure_lattice
                )
            
            # Add site to defects list
            defects_list.append(vacant_site)

            # Get atomic number change and defect type
            add_property={"original_an":ref_site.specie.Z,
                          "new_an": 0,
                          "an_change": 0 - ref_site.specie.Z,
                          "vacancy_defect": 1.0,
                          "substitution_defect": 0.0,
                          "bonds_broken": mindnn.get_cn(reference_struct, reference_struct.sites.index(ref_site)),
                          "site_fe": fe_site(ref_site.species_string,0)}
            defects_properties[vacant_site] = add_property

    # create a defects structure
    defects_struct = Structure.from_sites(defects_list)

    # Add properties to defects structure
    for a_site in defects_struct.sites:
        if a_site in defects_properties.keys():
            a_site.properties.update(defects_properties[a_site])
        else:
            pass

    return defects_struct

# Turn the defects structure into crystal graph
def get_c_graph(structure):
    sites_list = structure.sites

    # The nodes: These are the sites features
    nodes = []
    for i, site in enumerate(sites_list):
        node_features = [
            i, 
            site.properties["bonds_broken"], 
            site.properties["original_an"], 
            site.properties["new_an"], 
            site.properties["an_change"], 
            site.properties["vacancy_defect"], 
            site.properties["substitution_defect"], 
            site.properties["site_fe"]
        ]

        # Node features syntax
        nodes.append(node_features)
         

    # The edges
    edges = [] # The sites in relation
    edge_features = [] # The distance between each site

    from_e = []
    to_e = []
    
    for i, site_i in enumerate(sites_list):
        for j, site_j  in enumerate(sites_list):
            # Edges 
            from_e.append(i)
            to_e.append(j)

            # Get distance between sites
            dist = site_i.distance(site_j)

            # Are the defects the same or different
            same_diff = int(site_i.properties["an_change"] == site_j.properties["an_change"])

            # What is the site_fe difference
            site_fe_diff = np.abs(site_i.properties["site_fe"] - site_j.properties["site_fe"])

            edge_features.append([dist,same_diff,site_fe_diff])
            
    edges.append(from_e)
    edges.append(to_e)

    # The global features
    the_ids = []
    the_ratios = []
    total_sites = len(sites_list)

    the_formula = structure.formula
    composition = Composition(the_formula)
    element_dict = composition.get_el_amt_dict()

    for symb, numb in element_dict.items():
        try:
            ids = Element(symb).Z - 1
        except ValueError:
            ids = 0
        the_ids.append(ids)
        ration = numb/total_sites
        the_ratios.append(ration)

    return nodes, edges, edge_features, the_ids, the_ratios


def fast_graphy(dataset):
    graph_list = []
    unique_dataset_materials = dataset["dataset_material"].unique()


    for i in unique_dataset_materials:
        print(f"Working on {i}...")
        subset = dataset[dataset["dataset_material"] == i].reset_index(drop=True)

        pristine_structure = Structure.from_file(f"ref_cifs/{i}.cif")
        
        # Go through every row in the subset
        for _ , row in subset.iterrows():
            the_target = row["band_gap_value"]
            the_id = row["_id"]
            clean_defective_structure = Structure.from_file(f"original_dataset/{i}/cifs/{the_id}.cif")

            nodes, edges, edge_features, ids, ratios = get_c_graph(get_defects_structure(clean_defective_structure, pristine_structure))

            the_data = Data(
                x=torch.tensor(nodes, dtype=torch.float),
                edge_index=torch.tensor(edges, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                the_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0),
                the_ratios = torch.tensor(ratios, dtype=torch.float).unsqueeze(0),
                y=torch.tensor(the_target, dtype=torch.float).unsqueeze(0)
            )
                                    
            # Put all graphs together in a list
            graph_list.append(the_data)
            
    return graph_list


def main():
    comb_df = pd.read_csv("Final_Dataset/combined/combined_data.csv")
    train_set, val_set = train_test_split(comb_df, test_size=0.3, random_state=42, stratify=comb_df["strata"])
    val_set, test_set = train_test_split(val_set, test_size=0.5, random_state=42, stratify=val_set["strata"])

    full_train_graphs = fast_graphy(train_set)
    full_val_graphs = fast_graphy(val_set)
    full_test_graphs = fast_graphy(test_set)

    torch.save(full_train_graphs, "graphs/full_train_graphs.pt")
    torch.save(full_val_graphs, "graphs/full_val_graphs.pt")
    torch.save(full_test_graphs, "graphs/full_test_graphs.pt")


if __name__ == "__main__":
    main()