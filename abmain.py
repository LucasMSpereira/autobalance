import definitions as defs
from ax.service.ax_client import AxClient

trainSteps = 200_000
for leftCoef in [0.4, 0.6, 0.8, 1.0]:
    print(f'left coefficient: {leftCoef}', end = '    ')
    defs.functionDefs.explicitTest(trainSteps, leftCoef, -1, 15, True)

# Ax optimization
ax_client = AxClient() # create client
exp_params = [ # list of experiment parameters
        {
            "name": "dashLcoef",
            "type": "range",
            "value_type": "float",
            "bounds": [0.5, 1.0],
        },
        {
            "name": "dashRcoef",
            "type": "range",
            "value_type": "float",
            "bounds": [0.5, 1.0],
        },
        {
            "name": "trainSteps",
            "type": "fixed",
            "value_type": "int",
            "value": trainSteps,
        },
        {
            "name": "evalEps",
            "type": "fixed",
            "value_type": "int",
            "value": 70,
        },]
ax_client.create_experiment( # create experiment
    name = "Game auto balance", parameters = exp_params,
    # indicate which item is the objective in dictionary returned by evaluation function
    objective_name = "balance",
    minimize = True, outcome_constraints = ["episodes <= 1000000"])
defs.functionDefs.optLoop(10, ax_client, exp_params)