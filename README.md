# Weak-Neural-Variational-Inference_harmonic_elastography
Weak-Neural-Variational-Inference framework application for a synthetic problem in harmonic elastography.
The paper can be found open-access here: The full open-access paper can be found at: https://www.sciencedirect.com/science/article/pii/S0045782524007473
If you are looking for the base Weak-Neural-Variational-Inference framework with application in static elastography, you can find the GitHub page here: https://github.com/pkmtum/Weak-Neural-Variational-Inference

## Dependencies
- Python
- Fenics (install first and all following using pip)
- Fenics Adjoint
- torch with cuda
- scipy
- matplotlib
- tqdm

## Installation
Install Python and all dependencies mentioned above.
To clone this repo:
```
git clone https://github.com/pkmtum/Weak-Neural-Variational-Inference_harmonic_elastography.git
```

## How to run
We provide the code, input files, and example data to run the code. 
- Input.py can change code details like which data to load, where to calculate (GPU /CPU), posterior, etc.
- Generate_Ground_Truth_Fenics.py can generate ground truth displacement data that can be loaded instead of the data provided
- main.py executes the code
- 
In the current form, 24 GB of GPU memory (we used a Nvidia RTX 4090) was used.

## Citation
If this code is relevant to your research, we would be grateful if you cite our original work on the framework:
```
@article{scholz2024weak,
  title={Weak neural variational inference for solving Bayesian inverse problems without forward models: applications in elastography},
  author={Scholz, Vincent C and Zang, Yaohua and Koutsourelakis, Phaedon-Stelios},
  journal={arXiv preprint arXiv:2407.20697},
  year={2024}
}
```

## Contact
If you have questions or problems regarding the code or paper, please feel invited to reach out to us using the following E-Mails

Dipl.-Ing. Vincent C. Scholz:           vincent.scholz@tum.de

Dr. Yaohua Zang:                        yaohua.zang@tum.de
