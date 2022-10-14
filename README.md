# Multi-Granularity EHR Modeling

## Benchmark construction

   To support the comprehensive modeling of the *heterogeneity*, *irregularity*, and *interdependency* characteristics of clinical time series data, we build a new benchmark, named **ClinTS-HII**.
   This repository maintains all the documentation and needed scripts to build the ClinTS-HII benchmark.
   We selected 61 common biomarker variables to indicate the physiological states of a patient and 42 widely used interventions in intensive care units to represent the major events interdependent on these physiological states. 

   This benchmark includes a diverse set of clinical tasks covering different clinical scenarios for evaluation. The following table summarizes the statistics of these tasks.

   |  Task (Abbr.)   | Type  | # Train | # Val. | # Test | Clinical Scenario |
   |  :----  | :----: | ----: | ----: | ----: | :---- |
   | In-hospital Mortality (MOR)             | BC | 39, 449    | 4, 939  | 4, 970 | Early warning |
   | Decompensation (DEC)                    | BC | 249, 045    | 31, 896 | 30, 220 | Outcome pred. |
   | Length Of Stay (LOS)                    | MC | 249, 572   | 31, 970 | 30, 283 | Outcome pred. |
   | Next Timepoint Will Be Measured (WBM)   | ML | 223, 867   | 28, 754 | 27, 038 | Treatment recom. |
   | Clinical Intervention Prediction (CIP)  | MC | 223, 913   | 28, 069 | 27, 285 | Treatment recom. |


## Requirements

   Your local system should have the following executables:

   - [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   - Python 3.7 or later
   - git

## Download Data & Task Building

   ### I. Access to MIMIC-III data

   1. First you need to have an access to MIMIC-III Dataset, which can be requested [here](https://mimic.physionet.org/gettingstarted/access/). The database version we used here is v1.4.
   2. Download the MIMIC-III Clinical Database and place the MIMIC-III Clinical Database as either .csv or .csv.gz files somewhere on your local computer.


   ### II. Generate datasets

   Under construction... :construction:

## Run our model

   To run the model proposes in this paper, using the 

   ```bash
   python Main.py --task {task_name} \
                  --root_path {root_path} --data_path {root_path} \
                  --log {log_path} --save_path {save_path}
                  --epoch {epoch} --seed {seed} \
                  --lr 0.001 --batch {batch_size} \
                  --hie --adpt --dp_flag

   ```

   - ```task```: the downstram task name, select from ```[mor, decom, cip, wbm, los]```
   - ```hie```: do multi-granularity modeling or not
   - ```adpt```: do adaptive segmentation or not
   - ```dp_flag```: using DataParallel for training or not
   - ```seed```: the seed for parameter initialization.
   
   For more details, please refer to ```run.sh``` and ```Main.py```.

## License

The original [MIMIC-III database](https://mimic.mit.edu/docs/iii/) is hosted and maintained on [PhysioNet](https://physionet.org/about/) under [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciii/view-license/1.4/), and is publicly accessible at [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/).

Our code in this repository is licensed under the [MIT license](https://github.com/nullnullll/ClinTS_HII/blob/main/LICENSE).