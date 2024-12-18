# DataModelBias
This repository demonstrates how to analyze, identify, and mitigate bias in machine learning models using the Titanic dataset as a case study. The project focuses on understanding bias in data and model predictions, and implementing fairness-aware solutions to address these challenges.
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Bias in Data](#bias-in-data)
- [Model Training](#model-training)
- [Bias in Model Predictions](#bias-in-model-predictions)
- [Fairness Mitigation](#fairness-mitigation)
- [Getting Started](#getting-started)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
Machine learning models often inherit biases present in the data, leading to unfair outcomes. In this project, we:
- Explore biases in the Titanic dataset.
- Train a baseline model for survival prediction.
- Analyze fairness metrics in model predictions.
- Mitigate bias using the Fairlearn library.

This repository serves as a guide to integrating fairness considerations into your machine learning workflows.

---

## Dataset

The Titanic dataset contains information about passengers aboard the Titanic, including:
- Passenger class (`Pclass`)
- Gender (`Sex`)
- Age (`Age`)
- Number of siblings/spouses aboard (`Siblings/Spouses Aboard`)
- Number of parents/children aboard (`Parents/Children Aboard`)
- Fare (`Fare`)
- Survival status (`Survived`)

We introduce synthetic bias into the dataset by favoring female passengers over male passengers to illustrate bias mitigation techniques.

Dataset source: [Titanic Dataset](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv)

---

## Bias in Data
Bias can originate from historical, sampling, or societal inequities present in the data. In the Titanic dataset, potential biases include:
- Gender (`Sex`): Historically, women and children were given priority for survival.
- Class (`Pclass`): Higher-class passengers may have had better survival chances.

We analyze these biases and introduce a `Bias` feature to simulate a synthetic bias favoring female passengers.

---

## Model Training

### Baseline Model
We train a baseline RandomForestClassifier to predict survival status. The model is evaluated using standard metrics:
- Accuracy
- Confusion Matrix
- Classification Report

### Observations
While the baseline model achieves high accuracy, predictions may still reflect underlying biases from the data.

---

## Bias in Model Predictions
Bias in predictions is evaluated using fairness metrics:
- **Demographic Parity Difference**: Measures difference in positive prediction rates between groups.
- **Equalized Odds Difference**: Measures disparities in true positive and false positive rates between groups.

We use the Fairlearn library to calculate these metrics and identify disparities in model predictions.

---

## Fairness Mitigation

### Approach
To mitigate bias, we use `ExponentiatedGradient` from Fairlearn with `DemographicParity` constraints. This method adjusts model predictions to achieve fairer outcomes across sensitive groups.

### Results
The mitigated model demonstrates reduced bias while maintaining competitive predictive performance.

---

## Getting Started

### Prerequisites
- Python 3.7+
- Libraries: pandas, numpy, sklearn, fairlearn, matplotlib, seaborn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/data-model-bias-analysis.git
   cd data-model-bias-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Code
Execute the script to train models and analyze results:
```bash
python main.py
```

---

## Results

### Baseline vs Mitigated Model

| Metric                       | Baseline Model | Mitigated Model |
|------------------------------|----------------|-----------------|
| Accuracy                     | _value_        | _value_         |
| Demographic Parity Difference| _value_        | _value_         |
| Equalized Odds Difference    | _value_        | _value_         |

### Confusion Matrices
Visualization of confusion matrices before and after bias mitigation:

Baseline Model | Mitigated Model
:--------------:|:----------------:
![](images/baseline_cm.png) | ![](images/mitigated_cm.png)

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request explaining your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Fairlearn](https://fairlearn.org/) for fairness metrics and mitigation tools.
- [Scikit-learn](https://scikit-learn.org/) for model building and evaluation.

---

Feel free to fork and explore the repository to experiment with fairness-aware machine learning practices.
