# PyCoCo_templates
Building templates of ANY transient for which you have photomtry and sparse spectroscopy

References: https://arxiv.org/abs/1908.05228

You will need:
- Python3 and all the basics packages (numpy, scipy ..)
- The Pakcage george (version 0.3.1 at least, https://george.readthedocs.io/en/latest/)
## Build your own template: Instructions
Take a look at Figure 1 in the Paper.

![[Optional Text]](./pycoco_code_structure.pdf)

Each step of the process corresponds to a jupyter notebook in the folder ./Codes.

0. Clone this github repo and set the envoimental variable COCO_PATH to the path where you cloned the folder
i.e. export COCO_PATH="/Users/mariavincenzi/PyCoCo_templates/"
 
1. Prepare Inputs. Go into the folder /Inputs and prepare Photometry/Spectroscopy

