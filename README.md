# Science Scripts
Collection of Scripts for Handling Scientific Data

## XRD
1. **xrd_converter (GUI and CLI)**: Convert xrd data in complex csv or xrdml formats into simple csv format for plotting, or archiving. Bot GUI and command line versions available.
2. **xrd_plotter**: Plot one or more XRD data files (csv or xrdml format). Available both as a local python script (with GUI) or as a stand-alone web script (running either locally or in remote server).
    
    ### Functionality of the plotter:
    - Curve fitting
    - Curve normalization to peak
    - Cropping to view
    - Search of theoretical xrd based on formula or peak selection via Materials Genome (python version only).
    - Search of experimental xrd minerals based on peak selection [Rruff](https://www.rruff.net/zipped_data_files/powder/).
    - Smoothing curves
    - Background subtraction (via regularization or through reference diffractogram).
    - Correct angular offsets with reference data.
    
## Raman
1. **raman_plotter**: Plot one or more FTIR data files (H5, xml, or txt formats from Horiba Labspec). Available both as a local python script (with GUI, and ability the Rruff database  for Raman reference data) or as a stand-alone web script (running either locally or in remote server).
    
    ### Functionality of the plotter:
    - Curve fitting
    - Curve normalization to peak
    - Cropping to view
    - Search of experimental raman spectra of minerals based on peak selection [Rruff](https://www.rruff.net/zipped_data_files/raman/).
    - Search of theoretical raman based peak selection via Rruff. 
    - Smoothing curves
    - Background subtraction (via regularization or through reference spectra).
    
## FTIR
1. **ftir_plotter**: Plot one or more Raman data files (jdx or csv formats from Thermo Nicolet). Available both as a local python script (with GUI, and ability to access the Rruff databased for FTIR reference data) or as a stand-alone web script (running either locally or in remote server)
    
    ### Functionality of the plotter:
    - Curve fitting
    - Curve normalization to peak
    - Cropping to view
    - Search of experimental FTIR spectra of minerals based on peak selection [Rruff](https://www.rruff.net/zipped_data_files/infrared/).
    - Smoothing curves
    - Background subtraction (via regularization or through reference spectra).

## SEM/EDS
1. **Summarizer**: Create a summary given the spectra and the images. 

## PyMol Plotter
1. **plot_pymol_structures**: Plot a 'necklace' of spherical particles from a PACKMOL-style PDB file.

## Build GMSH
1. **Build_gmsh**: Bash script to compile [GMSH](https://gmsh.info/) for older system via miniconda3.
