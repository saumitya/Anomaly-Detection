# Anomaly-Detection
# README: Fault Detection Using Isolation Forest on the Tennessee Eastman Process Dataset

## Project Overview

This project implements an Isolation Forest-based machine learning model to detect faults in the Tennessee Eastman Process (TEP) dataset. The goal is to identify abnormal process behavior and classify data points as fault-free or faulty. The project uses Python for data preprocessing, model training, testing, and visualization of results.

## Repository Structure

- **Code Files**

  - `train_isolation_forest.py`: Contains the code to train the Isolation Forest model using the TEP dataset.
  - `test_isolation_forest.py`: Tests the trained Isolation Forest model on faulty data and visualizes predictions in real time.

- **Datasets**

  - `normalized_faulty_training.csv`: Preprocessed dataset containing faulty training data.
  - `normalized_faulty_testing.csv`: Preprocessed dataset containing faulty testing data.

- **Model Files**

  - `isolation_forest_model.pkl`: The trained Isolation Forest model saved in serialized format.

- **Documentation**

  - `Isolation_Forest_TEP_Report.docx`: Comprehensive report documenting the aim, abstract, literature review, methodology, and results of this project.

## Setup Instructions

### Prerequisites

- Python 3.10 or later
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

### Installation

1. Clone the repository or download the project files.
2. Install the required libraries by running the following command:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

### File Paths

Ensure the dataset files and trained model are placed in the appropriate paths as specified in the code files. Update file paths in the scripts if necessary.

## Usage

### Training the Model

1. Run `train_isolation_forest.py` to train the Isolation Forest model:
   ```bash
   python train_isolation_forest.py
   ```
2. The trained model will be saved as `isolation_forest_model.pkl`.

### Testing the Model

1. Run `test_isolation_forest.py` to test the model on faulty testing data and visualize predictions in real-time:

   ```bash
   python test_isolation_forest.py
   ```

2. The script will display two plots:

   - **Prediction Plot**: Displays whether the sample is predicted as fault-free (0) or faulty (1).
   - **Actual Fault Plot**: Displays the actual fault number for comparison.

## Key Features

- **Anomaly Detection**: The Isolation Forest algorithm identifies outliers in the dataset.
- **Real-Time Visualization**: Interactive plots provide real-time feedback on predictions and actual fault states.
- **Tennessee Eastman Process Dataset**: A widely-used benchmark dataset for fault detection and diagnosis.

## Results and Interpretation

- The model outputs predictions as either `0` (fault-free) or `1` (faulty).
- Real-time visualizations allow for intuitive monitoring of the model's performance.
- The detailed results and methodology are documented in `Isolation_Forest_TEP_Report.docx`.

## Acknowledgements

This project is based on the Tennessee Eastman Process dataset, a standard benchmark in process control and fault detection research. Special thanks to the research community for providing the dataset and inspiring methodologies for anomaly detection.

## Contact

For questions or further discussion, please contact **Saumitya Pareek at **[**saumityapareek@gmail.com**](mailto\:saumityapareek@gmail.com)** .**

