# [Pedestrian Crash Severity Analysis using TabNet and Stacking Ensemble Models](https://rdcu.be/dMrSv)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7194992.svg)](https://doi.org/10.1007/s42421-024-00098-x)
## Description
This repository contains the implementation of TabNet and stacking ensemble models aimed at analyzing pedestrian crash severity. By examining Utah's pedestrian crash data from 2010 to 2022, we apply these advanced models to identify key factors influencing crash outcomes. Our approach integrates generalized ordered probit, stacking ensemble models including XGBoost and extremely randomized trees, and the novel use of TabNet, enhanced with SHAP for interpretability. The analysis demonstrates TabNet's superior predictive accuracy and the stacking ensemble's robust performance in identifying critical crash severity determinants, such as pedestrian age, turning movements, lighting conditions, and alcohol consumption. This work contributes valuable insights into traffic safety engineering and policymaking to improve pedestrian safety.

## Methodology
- TabNet structure
![TabNet](https://github.com/pozapas/PedCrashAIM/blob/master/imgs/Pedsafety%20-%20TabNet.png)

- Stacking Ensemble Models structure
![SEM](https://github.com/pozapas/PedCrashAIM/blob/master/imgs/Pedsafety%20-%20Stacking.png)

## Installation
This project require Python and the following Python packages installed:
- scikit-learn
- XGBoost
- pandas
- imbalanced-learn
- SHAP
- matplotlib
- NumPy
- PyTorch
- PyTorch TabNet
- Optuna
- joblib
  
To install the Python packages, navigate to the local directory where you have cloned this repository and run the following command:
```bash
pip install -r requirements.txt
```
You need to have a `requirements.txt` file in your repository for this command to work. To create one, you can use the `pip freeze` command, which outputs all of the packages in the current environment. You can redirect this output to a file using the following command:
```bash
pip freeze > requirements.txt
```
Then you can manually edit `requirements.txt` to only include the necessary packages.
