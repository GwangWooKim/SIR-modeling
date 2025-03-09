<img src="/imgs/model.png" width="85%" height="85%">

# Application of [PINN](https://en.wikipedia.org/wiki/Physics-informed_neural_networks) to [SIR](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) modeling
This is an example of application of physics informed neural networks (PINNs). If you want to know more details, please refer to [`Final Report.pdf (Korean)`](https://github.com/GwangWooKim/SIR-modeling/blob/main/Final%20Report.pdf) in this repository that contains backgrounds, methods, results, and discussion.

## Dependency
* torch==1.13.1

## How to use
Just run the main.py!

## Results
<img src="/imgs/loss.png" width="70%" height="70%">
This means that my model was well optimized.
<img src="/imgs/SIR_PINN.png" width="70%" height="70%">
My PINN solves the system of ODEs and plots a figure like that.
<img src="/imgs/SIR_ODEsolver.png" width="70%" height="70%">
ODEsolver is the most useful tool of solving a system of ODEs. The tool plots a figure like that.
