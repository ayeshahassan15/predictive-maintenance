Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)



Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine – i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26


Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.



# Predictive Maintenance: Jet Engine Remaining Useful Life

Predicts how many operating cycles a jet engine has left before it fails,
using its sensor readings. Airlines and factories use this idea to plan
repairs before a breakdown instead of after one.

Data: NASA C-MAPSS turbofan degradation dataset, subset FD001
(100 training engines, 100 test engines, 21 sensors, 3 operating settings).

## Results

Final model: XGBoost.

| Evaluation | MAE | RMSE | R2 |
|---|---|---|---|
| GroupKFold on training engines | 13.22 | 18.41 | 0.805 |
| Official test set (last cycle of each engine) | 12.26 | 17.48 | 0.81 |

| Model | MAE (GroupKFold) |
|---|---|
| Linear Regression (baseline) | 17.43 |
| Random Forest | 13.25 |
| XGBoost | 13.22 |

On engines it has never seen, the model's RUL guess is off by about 12 cycles on average.

## Method

1. Created the RUL label per engine: last cycle minus current cycle.
2. Dropped 7 sensors that never change. 14 sensors kept.
3. Capped RUL at 125. Early in an engine's life the sensors look the same,
   so a capped label is easier to learn.
4. Built rolling mean and rolling std features (window of 5 cycles),
   calculated per engine and using only past data.
5. Validated with GroupKFold, so one engine is never in both train and validation.
6. Compared Linear Regression, Random Forest and XGBoost. Runs are tracked in MLflow.
7. Explained the model with SHAP.
8. Saved the model with joblib and served it with FastAPI.
9. Tested the data code and the API with Pytest.

## Findings

- Trees beat the linear baseline by about 4 cycles of MAE, because sensors
  rise in curves, not straight lines. Random Forest and XGBoost scored almost
  the same, so better features would likely help more than a fancier model.
- SHAP shows the model relies on rolling means of sensor_4, sensor_9,
  sensor_15, sensor_11 and sensor_14. High values of these sensors push the
  predicted RUL down, which matches the wear pattern seen in the sensor plots.
- The model is weakest in the middle of life (true RUL 50 to 100, MAE about 20),
  where the sensor signal is weak and noisy. Near failure the MAE is about 11.

## Limitations

- Simulated data, one dataset (FD001, a single operating condition).
- The test score uses only 100 engines, so it is noisy.
- MAE of about 11 near failure is still large for real maintenance decisions.

## Next steps

NASA asymmetric scoring function (late predictions cost more), feature
selection, prediction intervals, LightGBM, and FD002/FD004.

## Project structure

    data/raw/        raw C-MAPSS files (not in Git)
    src/             data.py, features.py, train.py, evaluate.py, ...
    api/             FastAPI app and Pydantic schemas
    models/          saved model (not in Git)
    tests/           Pytest tests

## How to run

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/save_model.py
uvicorn api.main:app --reload
pytest
```

Download C-MAPSS FD001 and place `train_FD001.txt`, `test_FD001.txt` and
`RUL_FD001.txt` in `data/raw/`.