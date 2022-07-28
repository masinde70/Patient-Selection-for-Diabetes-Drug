# Patient-Selection-for-Diabetes-Drug
Context: EHR data is becoming a key source of real-world evidence (RWE) for the pharmaceutical industry and regulators to make decisions on clinical trials. 
[The Data Visualisation Catalogue](https://datavizcatalogue.com/index.html)

#### Exploratory Data Analysis
The project correctly identified which field(s) has/have a high amount of missing/zero values.

The project correctly identified which field(s) has/have a Gaussian distribution shape based on the histogram analysis.

The project correctly identified fields with high cardinality.

The project justified why these fields had high cardinality.

The project correctly describes the distributions for age and gender.

Optional: The project uses Tensorflow Data Validation Visualizations to analyze which fields have a high amount of missing/null values or high cardinality.

✅ The exploratory data analysis was performed correctly - you identified the fields with missing and null values as well as columns with high cardinality, accurately described the distribution of age and gender and motivated your findings, great job!

#### Data Preparation

The project correctly identifies whether to include/exclude payer_code and weight fields.

The project justified why these fields should be included/excluded by using supporting data analysis.

✅ You nailed this part as well, having correctly chosen which fields (features like weight, payer_code, medical_specialty, and ndc_code) to remove and why.

✏️ For instance, the weight is really a useless field for this model and goes against the idea of the minimum necessary information.
The project uses the correct level(s) for the given EHR dataset (line, encounter, patient) and transforms, aggregates and filters appropriately.

✅ Nice - after the tests, the right Line level was used and appropriate filters, transformations, and aggregates are present in this submission.

The project correctly maps NDC codes to generic drug names and prints out the correct mappings in the notebook.

✅ Well done! The printed output shows that NDC Codes have been mapped correctly to generic drug names.

Tip:
✏️ As we observed, there are many codes that map to the same or similar drug. It's interesting to know that the same applies to other treatments as well - for example, primary and secondary procedures are often interchangeable and their codes can be modified whenever applicable; secondary procedures can be component codes of primary procedures, and health specialists can even determine themselves which code to assign to the procedure and whether it is a primary or secondary one as far as every single case is concerned.

Here's a random example of what I mean (check the "Coder Responsibility" section).

The project has correctly split the original dataset into train, validation, and test datasets.

The projects dataset splits do not contain patient or encounter data leakage.

The Projects code passes the Encounter Test.

✅ Solid checkmark - data has been split into train, validation, and test datasets correctly.

### Feature Engineering
The project correctly completes the categorical feature transformer boilerplate function.

The project successfully uses this function to transform the demo dataset with at least one new categorical feature.

✅ Reading the API, you have correctly transformed the categorical variables.

The project correctly completes the numerical feature transformer boilerplate function.

The project successfully uses this function to transform the demo dataset with at least one new numerical feature.

The project's transformer function correctly incorporates the provided z-score normalizer function for normalization or another custom normalizer.

✅ The results of your numerical feature transformation are also completely accurate (within the acceptable range).

#### Model Building and Analysis
The project has prepared the regression model predictions for TF Probability and binary classification outputs by doing the following:
Correctly utilized TF Probability to provide mean and standard deviation prediction outputs
Created an output prediction dataset that has the labels correctly mapped to a binary prediction and actual value.

Great job, your professionalism really shows off here!

The model has been evaluated across the following classification metrics: AUC, F1, precision, and recall.

Students have completed both questions for the model summary and address bias-variance tradeoff in regard to this problem.

✅ You have completed all the necessary questions and provided a brief, but detailed enough summary communicating your findings to a non-technical audience - to end-users who probably don't have much time and desire to dive into coding details (also, they might not have enough analytical proficiency for that).


The project contains a bias report with the following:

A visualization of at least two key metric(s) for patient selection
A visualization showing at least one reference group fairness example and its comparison on at least one metric (e.g. TPR).
Justification for analysis made about at least one visualization
✅ You created several unbiased, insightful, meaningful visualizations for your performance metrics and fairness examples.

✅ All the visuals are an accurate representation of the insight you are stating.

