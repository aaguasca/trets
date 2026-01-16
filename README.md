# TRETS
Temporal Resolution Estimator for Transient Sources (TRETS)

This package is a toolkit for probing transient emission with 
imaging atmospheric Cherenkov telescopes (IACTs). TRETS is build on top of Gammapy.

Currently, it only works for aperture-photometry IACT analyses.

## Install

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [anaconda](https://www.anaconda.com/distribution/#download-section) first. 

- Clone the repository
```
git clone https://github.com/aaguasca/trets.git
cd trets
```

- Install the environment
```
conda env create -f trets-environment.yml
pip install -e .
conda activate trets-dev
```

## License

TRETS is licenced under BSD 3-Clause License.