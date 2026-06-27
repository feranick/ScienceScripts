# Science Scripts
Collection of Scripts for Handling Scientific Data

## XRD
1. **xrd_converter (GUI and CLI)**: Convert xrd data in complex csv or xrdml format into simple csv format for plotting, or archiving. Bot GUI and command line versions available.
2. **xrd_plotter**: Plot one or more XRD data files (csv or xrdml format). Available both as a local python script (with GUI, and ability to access the Materials Genome datbase for XRD reference data) or as a stand-alone web script (running either locally or in remote server)
    
    ### Functionality of the plotter:
    - Curve fitting
    - Curve normalization to peak
    - Cropping to view
    - Search of theoretical xrd based on formula or peak selection via Materials Genome (python version only).
    - Smoothing curves
    - Background subtraction (via regularization or through reference diffractogram).
    - Correct angular offsets with reference data.

## SEM/EDS
1. **Summarizer**: Create a summary given the spectra and the images. 
