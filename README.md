# Patient-Selection-for-Diabetes-Drug

EHR data is becoming a key source of real-world evidence (RWE) for the pharmaceutical industry and regulators to make decisions on clinical trials. You are a data scientist for an exciting unicorn healthcare startup that has created a groundbreaking diabetes drug that is ready for clinical trial testing. It is a very unique and sensitive drug that requires administering the drug over at least 5-7 days of time in the hospital(X number of days based off of distribution that I will see in data and cutoff point) with frequent monitoring/testing and patient medication adherence training with a mobile application. You have been provided a patient dataset from a client partner and are tasked with building a predictive model that can identify which type of patients the company should focus their efforts testing this drug on. Target patients are people that are likely to be in the hospital for this duration of time and will not incur significant additional costs for administering this drug to the patient and monitoring.

##

Expected Hospitalization Time Regression and Uncertainty Estimation Model: Utilizing a synthetic dataset(upsampled, denormalized, with line level augmentation) built off of the UCI Diabetes readmission dataset, students will build a regression model that predicts the expected days of hospitalization time and an uncertainty range estimation.

###

This project will demonstrate the importance of building the right data representation at the encounter level, with appropriate filtering and preprocessing/feature engineering of key medical code sets. we analyze and interpret the model for biases across key demographic groups. Lastly, we will utilize the TF probability library to provide uncertainty range estimates in the regression output predictions to prioritize and triage prediction uncertainty levels.


###
