# PyCoCo_templates
Building templates of ANY transient for which you have photomtry and sparse spectroscopy

References: https://arxiv.org/abs/1908.05228

You will need:
- Python3 and all the basics packages (numpy, scipy ..)
- The Pakcage george (version 0.3.1 at least, https://george.readthedocs.io/en/latest/)

### Build your own template: Instructions

**STEP 0**: Clone this github repo and set the envoimental variable COCO_PATH to the path where you cloned the folder
i.e. export COCO_PATH="/Users/mariavincenzi/PyCoCo_templates/"

**STEP 1**: Prepare Inputs.

All the inputs (Photometry, Spectroscopy, Other info about the transient like Galactic/host extinction, Redshift...), Filter transmission function) should be placed in ./Inputs

You essentially need to do 3 things before you start running the code:

1) In the folder `./Inputs/Photometry/0_LCs_mags_raw` place your photometry in magnitudes. See example to check the column format. If your photometry is already in flux or is already dust corrected (or you don’t want to dust correct it) and/or is already extended at early/late time (or you don’t want to extend it at early/late times) just skip all these step, do not run the first 4 notebooks and place the photometry directly in `./Inputs/Photometry/4_LCs_late_extrapolated`
2) In the folder `./Inputs/Spectroscopy/1_spec_lists_original` place for each template a file with the list of the spectra. See example to check the column format and file name. In the folder `./Inputs/Spectroscopy/1_spec_original` place the actual files with the spectra. See example to check the format of the spectra. If you want to smooth the spectra use the provided notebook. Otherwise skip this and put list and spectra in the folders `./Inputs/Spectroscopy/2_spec_lists_smoothed` and `./Inputs/Spectroscopy/2_spec_smoothed` .
3) Modify file `./Inputs/SNe_info/info.dat` and add a row for each new template you want to build.

See example provided to see the format all these info should be provided.

4) (optional) In the folder `./Inputs/2DIM_priors` you find the surface prior for Stripped Envelope SNe, Hydrogen-rich SNe (basically Type II)
 and IIn SNe. If you want to use another prior for the 2-dim GP modelling, add it in the `./Inputs/2DIM_priors` folder  and specify which prior file
 you want to use in the notebook `./Codes/6_TwoDim_UVExtend_Extrapolate.ipynb`

All the outputs (LC fit, mangled spectra, various plots and final template) will be created in Outputs.

Take a look at Figure 1 in the Paper.
Each step of the process corresponds to a jupyter notebook in the folder ./Codes (I found debugging and output visualizaztion easier using notebooks...)
![Imgur](pycoco_code_structure.png)

Usually each notebook is structured as follow:
- 1st cell(s): define PATHs and load modules
- 2nd cell: initialize a Class and all the related functions
- 3rd cell(s): build the class and run the actual code and save the output
