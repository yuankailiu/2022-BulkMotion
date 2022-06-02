# Codes, docs, and Jupyter Notebooks for:

Oliver L. Stephenson, Yuan-Kai Liu, Zhang Yunjun, Mark Simons, and Paul Rosen (2022). The Impact of Plate Motions on Long-Wavelength InSAR-Derived Velocity Fields. *Geophysical Research Letters [Paper # 2022GL099835]*.


## Folder structure:

### codes
Mostly plotting utility functions that are called in the notebooks.

### inputs
+ JSON files that contains the original file paths.
+ A table of NNR MORVEL 56 plate motion model (not been used here; only for future reference)

### notebooks
Jupyter Notebooks for plotting

NOTE: These notebook is based on the released version of MintPy-1.3.3 and NOT maintained for furture development.
+ [Fig. insets](https://nbviewer.org/github/yuankailiu/2022-BulkMotion/blob/main/notebooks/Fig_insets_comparePMMs.ipynb) - Plate motion model insets showing the plate geometry and the motion quivers
+ [Fig. 2 and S1-S5](https://nbviewer.org/github/yuankailiu/2022-BulkMotion/blob/main/notebooks/Fig2_S1-S5_correction_series.ipynb) - A suite of corrections applied in Makran, Aqaba, and Australia regions (both ascending and descending)
+ [Fig. 3+4](https://nbviewer.org/github/yuankailiu/2022-BulkMotion/blob/main/notebooks/Fig3-4_separate.ipynb) - Evaluate the impact of plate motion model correction on Makran, Aqaba, and Australia regions (plot velocity maps, range ramps)
+ [Fig. S6](https://nbviewer.org/github/yuankailiu/2022-BulkMotion/blob/main/notebooks/FigS6_roughEstimate.ipynb) - Global impact of the range ramp due to plate motion. This's an rough estimate using the Aqaba LOS geometry
+ [Fig. S7](https://nbviewer.org/github/yuankailiu/2022-BulkMotion/blob/main/notebooks/FigS7_asc_dsc2hv.ipynb) - Quasi-vertical and -horizontal decomposition of the velocity field using both ascending and descending velocities
+ other notebooks that are not used in the paper directly
