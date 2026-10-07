# Exploring the Band Gap of 2D Crystalline Materials With Point Defects Using a Machine Learning-based Model
This repository is set up to build an original GNN machine learning model that can accurately predict the band gap of 2D crystalline materials with vacancy or substitutional point defects.

## The Data
The materials involved in this dataset are:
1. Boron nitride ($BN$)
2. Black phosphorus ($bP$)
3. Gallium selenide ($GaSe$)
4. Indium selenide ($InSe$)
5. Molybdenum disulfide ($MoS_2$)
6. Tungsten diselenide ($WSe_2$)

These materials may have the vacancy or substitution defects at different concentrations and arranged in a variety of ways. They can have a low concentration of defects (1, 2, or 3 defect sites) or a high concentration of defects(2.5%, 5%, 7.5%, 10%, or 12.5% of the total sites are defect sites).

The dataset with low concentration of defects has 5933 datapoints involving $WSe_2$ and $MoS_2$ materials totaling 11866. This dataset prioritizes the vast possibilities of arrangement of defect sites in a material.

For the highly concentrated dataset, all the materials mentioned above have 100 datapoints with 2.5%, 5%, 7.5%, 10%, and 12.5% defect sites. This makes up 500 datapoints for each of the 6 materials totaling 3000 datapoints. This dataset prioritizes the concentration of defect sites in a material.

The model learns to identify how the type of host material, the type of defect present, the concentration of the defects in the material and the arrangement of the said defects in the material affects the bandgp of the material and inturn predict the band gap of the defective material.

## Execution

To ilustrate the model in action, you will need:
1. The pristine structure in cif format. 
2. The subsequent defective structure in cif format. 

Note that the model was built on the materials mentioned above and with vacancy and substitution defects only and therefore different materials and defect types will result to an inaccurate prediction if even possible.

With these 2 structures, you could run:

```
python demo.py --p <file/path/to/pristine.cif> --d <file/path/to/defective.cif>
```

or use `demo.ipynb` and add the path to the pristine and defective structures