
# Summary
This repository supports the [Europython 2024](https://ep2024.europython.eu/session/one-analysis-a-day-keeps-anomalies-away) talk "One analysis a day keeps anomalies away!" focused on the **unsupervised analysis of multivariate time series data**. 

The goal is to provide an extensive overview of various approaches and techniques, along with a hands-on starting point for practitioners to quickly benchmark these methods on their datasets.


## Introduction
In this talk, we will explore different methods for unsupervised anomaly detection on multivariate timeseries data. We will discuss the challenges associated with anomaly detection and how these approaches can help in identifying anomalies in real-world datasets.  

You can find the presented slides [here](#introduction)

## Datasets
To benchmark your results, you can start with [ADBench](https://github.com/Minqi824/ADBench), a collection of 57 curated datasets.  

### Using your dataset
If you prefer to use your own dataset, save it under the **data** folder and add loading functionality to *data_utils.py* (i.e. under *get_data* method). 

The *get_data* method is written to support the format of ADBench, assuming a .npz input file file contains 2 matrices: 
- **X** a <n_timesteps> * <n_timeseries> matrix without gaps and ordered by time (implicit time index) 
- **y** a <n_timesteps> elements vector, containing the anomaly annotations, a binary vector where 1 represents anomaly present. 

Since this repository addresses anomaly detection in an **unsupervised** setting, annotations are used only for evaluation, not for training.

Note: If your dataset has a different format, you must adapt it to the above requirements (e.g., within the get_data method).


## Evaluation
The experimental results can be evaluated with the functionality in *evaluation_utils.py*.  
Key methods include:
- *run_evaluation*: evaluates the **correctness** of the results by computing the AUCROC, AUCCPR, F1, Precision and recall scores. The method computes the same scores also after performing **point adjustement**.
- *threshold_anomalies*: performs **non parametric dynamic thresholding** using various algorithms from [pythresh](https://pythresh.readthedocs.io/en/latest/index.html).


## Environment
Some of the used libraries depend on a **fixed python version**. Run the script below to create the required repository.
```
conda create -y -n adenv python=3.9
conda activate adenv
pip install -r requirements.txt
```

## Approach 1: Traditional AD methods
This section covers the baseline approaches to implement unsupervised anomaly detection. The methods in this category are usually applied to tabular data.
- *sklearn.ipynb* contains examples of performing anomaly detection using [sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html) methods
- *pyOD.ipynb* contains examples of the dedicated library [pyOD](https://pyod.readthedocs.io/en/latest/index.html)


## Approach 2: Timeseries based methods
This section covers approaches analyzing the temporal component of time-series data
- *darts.ipynb*  exemplifies the usage of [darts library](https://unit8co.github.io/darts/) for timeseries forecasting/reconstruction. After predicting the forecast/reconstruction the package pythresh is used to find optimal thresholds and predict anomalies based on prediction errors.
- *tsfresh.ipynb* illustrates the approach of transforming temporal data into tabular data by feature extraction. Using the library [TSFresh](https://tsfresh.readthedocs.io/en/latest/index.html) we can extract temportal features from overlapping rolling windows.


## Approach 3: Real-time/streaming approach
This section covers approaches focused on addressing the real-time scenario
- *pySAD* exemplifies solving anomaly detection for streaming, using [pySAD](https://pysad.readthedocs.io/en/latest/api.html#module-pysad.core) library and state of the art algorithms like LODA and xStream


## Approach 4: Auto ML
This section covers approaches leveraging auto ML to find optimal models
- *pycaret* exemplifies the usage og the anomaly detection functionality from autoML [pycaret](https://www.pycaret.org/tutorials/html/ANO101.html) 


## Approach 5: Latest published cutting-edge methods 
This section covers approaches leveraging the latest state-of-the-art anomaly detection methods using deep learning approaches:
- *deepOD* exemplifies the usage of the [deepOD](https://deepod.readthedocs.io/en/latest/index.html) containing reconstruction-, representation-learning-, and self-superivsed-based latest deep learning contributions

## Evaluation
All methods presented above evaluate the experimental results using traditional scores (F1, AUC, Precision, Recall) as well as **point adjustment** counterparts. 

After running each notebook, the summary of results is saved in a dictionary under the results folder.

## Benchmarking
All notebooks above save a dump of performance results to a file under **results** repositories.  
To compare all experimental methods, run the notebook **benchmark.ipynb**.  
A unified table of results is produced together with comparative visualizations.

## Conclusion
Despite the numerous existing approaches, unsupervised anomaly detection remains a challenging field. There is no silver bullet - each dataset brings its own challenges and idiosynchrasies.

Feel free to explore the code and examples provided in this repository. We hope this resource will be helpful in understanding and implementing unsupervised anomaly detection techniques on your own datasets.

Happy coding!