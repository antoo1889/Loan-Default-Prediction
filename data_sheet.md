# Datasheet Template

Datasheet version: v0.1

## Motivation 

The loan_default_dataset.csv was created by Coursera as part of the course Data Science Coding Challenge: Loan Default Prediction. I access the public domain data published in 20223 in Kaggle.
The data was created to develop ML models to tackle one relevant financial problem that concers the majority of the financial institutions, loans defaults. 

One of the primary objectives of companies with financial loan services is to reduced payment defaults and ensure that individuals are paying back their loans as expected. Having a considerable large data set can help ML developments to decide what models it the best fit for it. It corresponds to a synthetic dataset built to simulate real-world credit risk scenarios.

Based on the research made, the dataset was not collected from real users or financial institutions. Instead, it was generated artificially using simulated data to resemble real-world patterns in loan applications and defaults, with the goal of providing a safe, anonymized, and educational dataset that mimics the structure of real loan data.

## Composition

The dataset contains 255,347 rows and 18 columns in total.

![Image](images/data_image1.png)

Each instance in the dataset represents a loan application made by a single individual. The features that describe the instances are:

*   Demographics: Age, Education, MaritalStatus, HasDependents
*   Employment and Income: EmploymentType, Income, MonthsEmployed
*   Credit Behavior: CreditScore, NumCreditLines, HasMortgage, 
*   Loan Characteristics: LoanAmount, InterestRate, LoanTerm, DTIRatio, LoanPurpose, HasCoSigner
*   Target variable: Default (binary classification: 0 = no default, 1 = default)

The distribution of the instances for each type is:

Absolute:

Non-Default  225.694
Default 29.653

Proportional:

Non- Default0.883872
Default 0.116128

There is no missing data in the data set. 

If the dataset was real world data, it would contain confidential information about employment, income and credit behaviour for individual that solicitated the loan, however, as mentioned the data is synthetic what means that it was not collected from real users. 

## Collection process

The dataset was not collected from real users or financial institutions. Instead, it was generated artificially using simulated data to resemble real-world patterns in loan applications and defaults, with the goal of providing a safe, anonymized, and educational dataset that mimics the structure of real loan data.

## Preprocessing/cleaning/labelling

There was not cleaning process I could find. I did my own analysis and cleaning. 

## Uses

The dataset is quite imbalanced, as mentioned before (Non defaulted observations 88% vs defaulted observations 12%). This actually reflects real-world scenarios where default rates are generally < 10–15%. However, imbalanced classes can cause the model to:

* Be biased toward the majority class.
* Fail to detect important rare events, defaults in this case
* Show misleading metrics like high accuracy, even if it's not learning anything useful

The dataset potential goes futher than only loans defaults prediction, it could be used for:

- Clustering borrower profiles, using unsupervised learning (K-Means) to segment applicants into low-income high-risk, high-income low-risk group, for example. Or create groups based on age. 
- Anomaly detection, aiming to find unusual applications (e.g., suspiciously high income with bad credit). Could be useful for fraud detection or data quality checks.
- Another option could be loan amount prediction, based on features like income, credit score, employment, etc. This could contribute to understand lending patterns or setting policy limits.

On the other hand, the data set has some limitations for the following problems:

- Time series or trend analysis since there is no timeline information about the data. This restricts the analysis for analysis over time and economic events consideration like recessions that could affect loans performance. 
- Behaviours analysis after the loan was issued and overtime, considering that the dataset contains information at the time of the application only. 

## Distribution
Published in Kaggle with license CC0: Public Domain.

## Maintenance
There’s no maintenance of the data.
