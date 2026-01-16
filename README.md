# TRETS
Temporal Resolution Estimator for Transient Sources (TRETS)

This package is a toolkit for probing transient emission with 
imaging atmospheric Cherenkov telescopes (IACTs). TRETS is build on top of Gammapy.

Currently, it only works for aperture-photometry IACT analyses.

## Dependencies

- gammapy >= 2.0

## Install

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [anaconda](https://www.anaconda.com/distribution/#download-section) first. 

## Install for users

- Follow the installation of [gammapy](https://docs.gammapy.org/stable) to install gammapy >= 2.0.

- Clone the repository and install the Package.
```
git clone https://github.com/aaguasca/trets.git
cd trets
pip install .
```

## Install for developers

- Clone the repository
```
git clone https://github.com/aaguasca/trets.git
cd trets
```

- Install the environment
```
conda env create -f trets-environment.yml
pip install -e .
pre-commit install
```

- Activate the environment
```
conda activate trets-dev
```

## How to contribute
See documentation on gammapy.

## License

TRETS is licenced under BSD 3-Clause License.