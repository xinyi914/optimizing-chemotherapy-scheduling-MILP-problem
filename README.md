# optimizing-chemotherapy-scheduling-MILP-problem
This project is optimizing chemotherapy scheduling. There are three approaches. The first is based on the paper "Optimal override policy for chemotherapy scheduling template via mixed-integer linear programming". This approach only decide which policy each time slot uses. The second approach is an integrated model to decide the start time and end time by minimizing the override policy. Both approaches are implemented in deterministic case and stochastic case. The third approach is a two step method for the stochastic case of second approach due to the long computation time of the second approach. More details can be found in "result slide.pptx"

Data are generated in the folder "data".

Model 1 and model 2 are writted in jupyter notebook. Model3 is organized in main.py.

More details about model 3 can be found in "model3_presentation.pptx"
