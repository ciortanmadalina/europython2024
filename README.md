
# Summary
This repository is created for a talk at [Europython 2024](https://ep2024.europython.eu/session/one-analysis-a-day-keeps-anomalies-away) to showcase different approaches to implement unsupervised anomaly detection on multivariate timeseries data. The repository provides an overview of various approaches and techniques.


## Introduction
In this talk, we will explore different methods for unsupervised anomaly detection on multivariate timeseries data. We will discuss the challenges associated with anomaly detection and how these approaches can help in identifying anomalies in real-world datasets.  

You can find the presented slides [here](#introduction)

## Datasets
A starting point to benchmark results is [ADBench](https://github.com/Minqi824/ADBench), a collection of 57 curated datasets.  

If you want to use your dataset, save it under the **data** folder and add loading functionality to *data_utils.py* (i.e. under *get_data* method). 


## Evaluation
The experimental results can be evaluated with the functionality in *evaluation_utils.py*. Key methods:
- *run_evaluation*: evaluates the **correctness** of the results by computing the AUCROC, AUCCPR, F1, Precision and recall scores. The method computes the same scores also after performing **point adjustement**.
- *threshold_anomalies*: performs **non parametric dynamic thresholding** using various algorithms from [pythresh](https://pythresh.readthedocs.io/en/latest/index.html).


## Environment
Some of the used libraries depend on a fixed python version. Run the script below to create the required repository.
```
conda create -y -n adenv python=3.9
conda activate adenv
pip install -r requirements.txt
```

## Approach 1: Traditional AD methods
This section will cover the first approach to implement unsupervised anomaly detection. We will discuss the underlying algorithm, its advantages, and limitations. Code examples and visualizations will be provided to demonstrate the approach.

## Approach 2
In this section, we will explore another approach to implement unsupervised anomaly detection. We will discuss the algorithm, its applicability to multivariate timeseries data, and any specific considerations. Code snippets and results will be shared to illustrate the approach.

## Approach 3
The third approach focuses on a different technique for unsupervised anomaly detection. We will delve into the details of the algorithm, its strengths, and potential use cases. Code samples and visualizations will be provided to enhance understanding.

## Conclusion
In the final section, we will summarize the key takeaways from the talk. We will discuss the pros and cons of each approach, highlight any challenges faced, and provide recommendations for implementing unsupervised anomaly detection on multivariate timeseries data.

Feel free to explore the code and examples provided in this repository. We hope this resource will be helpful in understanding and implementing unsupervised anomaly detection techniques on your own datasets.

Happy coding!