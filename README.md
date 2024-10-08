# SimNIBS

![SimNIBS Fronpage](docs/_static/gallery/simnibs_workflow.png)
The main goal of SimNIBS is to calculate electric fields caused by Transcranial Electrical Stimulation (TES) and Transcranial Magnetic Stimulation (TMS).
 
The pipeline is divided in three parts:
1. Automatic segmentation of MRI images and meshing to create individualized head models
2. Calculation of electric fields through the Finite Element Method (FEM)
3. Post-processing of results for further analysis.


## Build Status
| Linux   | Windows    | MacOS |
|---------|------------|-----|
| [![Build Status](https://dev.azure.com/simnibs/simnibs/_apis/build/status/Linux?branchName=master)](https://dev.azure.com/simnibs/simnibs/_build/latest?definitionId=4&branchName=master) | [![Build Status](https://dev.azure.com/simnibs/simnibs/_apis/build/status/Windows?branchName=master)](https://dev.azure.com/simnibs/simnibs/_build/latest?definitionId=5&branchName=master) |  [![Build Status](https://dev.azure.com/simnibs/simnibs/_apis/build/status/MacOS?branchName=master)](https://dev.azure.com/simnibs/simnibs/_build/latest?definitionId=9&branchName=master)   |

## Getting Started

SimNIBS runs on 64bit Windows, Linux and MacOS machines.
Please visit [the SimNIBS website](https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html) for instructions on how to download and install SimNIBS.

## Installation for development

After cloning the repository (for all platforms except arm64 Macs):

```
conda env create -f environment_.yml -n <name_of_my_environment>
conda activate <name_of_my_environment>
python -m pip install --editable .
python simnibs/cli/link_external_progs.py
```


### Rawan Installation Guide
If above didnt work, try the following:
```
conda env create -f "environment_win.yml"
conda activate simnibs_env
pip install -f https://github.com/simnibs/simnibs/releases/latest simnibs
```

If this not working use, download the `.whl` file then:
```
pip install simnibs-4.1.0-cp39-cp39-win_amd64.whl --user
```
If loop module import raise error, try to delete `simnibs\__init__.py` file.

> #### Example dataset
> Example dataset can be downloaded from the [following link](https://simnibs.github.io/simnibs/build/html/dataset.html):
> https://github.com/simnibs/example-dataset/releases/latest/download/simnibs4_examples.zip
 
> **NOTE**
>
> Currently on arm64 Macs it is only possible to build by creating an x86_64 environment. To do so, replace the first command above with:
> ```
> conda env create -f environment_.yml -n <name_of_my_environment> --platform osx-64
> ```
> If the `--platform` argument is not available in your version of `conda`, use
> ```
> CONDA_SUBDIR=osx-64 conda env create -f environment_.yml -n <name_of_my_environment>
> conda env config vars set CONDA_SUBDIR=osx-64 -n <name_of_my_environment>
> ```
> The latter command ensures that new packages are also installed from the `osx-64` subdir in this environment.

## Authors
Please see [the SimNIBS website](./docs/contributors.rst) for a complete list of contributors.

## 3rd Party Files
We have included code or binaries from the following project to this repository:
* [Gmsh](https://www.gmsh.info)
* [meshfix](https://github.com/MarcoAttene/MeshFix-V2.1)
* [CAT12](http://www.neuro.uni-jena.de/cat/)
* [PETSc](https://www.mcs.anl.gov/petsc/)
* [HYPRE](https://github.com/hypre-space/hypre)
* [MPICH](https://www.mpich.org/)
* [MSMPI](https://github.com/Microsoft/Microsoft-MPI)
* [CYGWIN](https://www.cygwin.com/)
* [pygpc](https://github.com/konstantinweise/pygpc)
* [PyPardiso](https://github.com/haasad/PyPardisoProject)
* [CGAL](https://www.cgal.org/)
* [Mmg](https://www.mmgtools.org/)
* [FMM3D](https://github.com/flatironinstitute/FMM3D)

For a full list of files and licenses, please see the [3RD-PARTY.md](3RD-PARTY.md) file
