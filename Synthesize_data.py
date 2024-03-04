# -*- coding: utf-8 -*-
"""
Based on the original file located at
    https://colab.research.google.com/drive/1F3WWduNjcX4oKck6XkjlwZ9zIsWlTGEM

If we stop considering binary inputs we can use this script to synthetize data : 
    - Create some real data with the real concentrations and we assign an infection risk
    - Use this adapted and functionnal script to synthetize more data
    - Train a more complicated NN than a single layer perceptron
"""
### Import the SDV (synthetic data vault) library
import sdv
from sdv.lite import SingleTablePreset
import pandas as pd

### 1. Creating pre-existing data
"""
For now the data is only binary because it was a first approach but it is not possible to create more binary
inputs than below
"""
X_bin = [[0, 0, 0,	0],
    [1,	0,	0,	0],
    [0,	1,	0,	0],
    [0,	0,	1,	0],
    [0,	0,	0,	1],
    [1,	1,	0,	0],
    [1,	0,	1,	0],
    [1,	0,	0,	1],
    [0,	1,	1,	0],
    [0,	1,	0,	1],
    [0,	0,	1,	1],
    [1,	1,	1,	0],
    [1,	1,	0,	1],
    [1,	0,	1,	1],
    [0,	1,	1,	1],
    [1,	1,	1,	1]]

X_float = [[0, 0, 0, 0],
    [1,	1,	1,	1],
    [0.45, 0, 0, 0],
    [0,	0.75,	0,	0],
    [0,	0,	0.65,	0],
    [0,	0,	0,	0.85],
    [0.65,	0.25,	0,	0],
    [0.3,	0,	0.75,	0],
    [0.45,	0,	0,	0.65],
    [0,	0.65,	0.85,	0],
    [0,	0.9,	0,	0.55],
    [0,	0,	0.7,	0.6],
    [0.55,	0.8,	0.9,	0],
    [0.25,	0.75,	0,	0.8],
    [0.95,	0,	0.75,	0.55],
    [0,	0.35,	0.45,	0.6],
    [0.8,	0.55,	0.65,	0.95],
    [0,	0.45,	0.8,	0.2],
    [0.2,	0.75,	0.95,	0.45],
    [0.85,	0.55,	0.25,	0.25],
    [0.25,	0.65,	0.10,	0.55],
    [0.60,	0.7,	0.95,	0.85]]


Y_cat_bin = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]
Y_cat_float = [0, 4, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 3, 2, 4, 2, 3, 3, 2, 4]

real_data = pd.DataFrame(X_float, columns=["AI-2", "Urea", "Toxins", "Siderophores"])
real_data.insert(4,"Risk", Y_cat_float)
real_data.insert(0,"Id", range(len(Y_cat_float)))


#print(real_data.head())

### 2. Creating a synthesizer
"""
An SDV **synthesizer** is an object that you can use to create synthetic data. 
It learns patterns from the real data and replicates them to generate synthetic data.

Let's use the `FAST_ML` preset synthesizer, which is optimized for performance.
"""

# The metadata is not of the type df.info(), it is class of its own 
metadata = sdv.metadata.SingleTableMetadata()
metadata.METADATA_SPEC_VERSION = "SINGLE_TABLE_V1"
metadata.columns = {
        "Id": {
            "sdtype": "id"
        },
        "AI-2": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Urea": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Toxins": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Siderophores": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Risk": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        }
        }
metadata.primary_key = "Id"


synthesizer = SingleTablePreset(
    metadata,
    name='FAST_ML'
)

"""Next, we can **train** the synthesizer. We pass in the real data so it can learn patterns using machine learning."""

synthesizer.fit(
    data=real_data
)


### 3. Generating synthetic data
"""
Use the `sample` function and pass in any number of rows to synthesize.
"""

synthetic_data = synthesizer.sample(
    num_rows=50
)

print(synthetic_data.head())


### 4. Evaluating real vs. synthetic data
"""
SDV has built-in functions for evaluating the synthetic data and getting more insight.
"""

## 4.1 Diagnostic
"""
As a first step, we can run a **diagnostic** to ensure that the data is valid. SDV's diagnostic performs some basic checks such as:

- All primary keys must be unique
- Continuous values must adhere to the min/max of the real data
- Discrete columns (non-PII) must have the same categories as the real data
- Etc.
"""

from sdv.evaluation.single_table import run_diagnostic

diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)


### 4.2 Data Quality
"""
We can also measure the **data quality** or the statistical similarity between the real and synthetic data. 
This value may vary anywhere from 0 to 100%.
"""

from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

"""
According to the score, the synthetic data is about 87% similar to the real data in terms of statistical similarity.

We can also get more details from the report. For example, the Column Shapes sub-score is 92%. Which columns had the highest vs. the lowest scores?
"""

quality_report.get_details('Column Shapes')


### 4.4 Visualizing the data
"""
For even more insight, we can visualize the real vs. synthetic data.

Let's perform a 1D visualization comparing a column of the real data to the synthetic data.
"""

from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='Toxins',
    metadata=metadata
)

fig.show()

"""We can also visualize in 2D, comparing the correlations of a pair of columns."""

from sdv.evaluation.single_table import get_column_pair_plot

fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['AI-2', 'Risk'],
    metadata=metadata
)

fig.show()

fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['Urea', 'Risk'],
    metadata=metadata
)

fig.show()

fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['Toxins', 'Risk'],
    metadata=metadata
)

fig.show()

fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['Siderophores', 'Risk'],
    metadata=metadata
)

fig.show()

### 5. Saving and Loading
"""
We can save the synthesizer to share with others and sample more synthetic data in the future.
"""

# synthesizer.save('my_synthesizer.pkl')

# synthesizer = SingleTablePreset.load('my_synthesizer.pkl')
 