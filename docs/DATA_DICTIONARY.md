# Data Dictionary

This document provides detailed definitions and descriptions of all variables used in the study.

## Dataset Overview

- **Source**: Korean peer-to-peer (P2P) lending platform
- **Time period**: 2006–2016
- **Sample size**: 6,057 loan applications
- **Target variable**: Binary default indicator (1 = default, 0 = repayment)
- **Default rate**: 55.34% (3,352 defaults out of 6,057 loans)

## Structured Variables (11)

### 1. Gender

- **Type**: Binary categorical
- **Encoding**: 0 = Male, 1 = Female
- **Description**: Borrower's gender
- **Statistics**:
  - Mean: 0.33
  - Interpretation: 33% female, 67% male

### 2. Region

- **Type**: Binary categorical
- **Encoding**: 0 = Metropolitan area, 1 = Non-metropolitan area
- **Description**: Borrower's residential region
- **Statistics**:
  - Mean: 0.52
  - Interpretation: 52% non-metropolitan, 48% metropolitan

### 3. Age

- **Type**: Continuous (integer)
- **Unit**: Years
- **Description**: Borrower's age at the time of loan application
- **Statistics**:
  - Mean: 36 years
  - Std Dev: 6.41 years
  - Median: 35 years
  - Range: 21–68 years
  - Q1: 32 years, Q3: 40 years

### 4. Credit Score

- **Type**: Continuous (integer)
- **Unit**: Points (0–1,000 scale)
- **Description**: Borrower's credit score from credit bureau
- **Statistics**:
  - Mean: 629
  - Std Dev: 129.4
  - Median: 638
  - Range: 0–950
  - Q1: 577, Q3: 706
- **Note**: 0 indicates missing or unavailable credit score

### 5. Monthly Income

- **Type**: Continuous
- **Unit**: 10,000 KRW (Korean Won)
- **Description**: Borrower's self-reported monthly income
- **Statistics**:
  - Mean: 261 (2,610,000 KRW)
  - Std Dev: 228.9
  - Median: 200 (2,000,000 KRW)
  - Range: 0–6,000 (0–60,000,000 KRW)
  - Q1: 160, Q3: 300
- **Note**: 0 indicates missing or unemployed

### 6. Loan Amount

- **Type**: Continuous
- **Unit**: 10,000 KRW (Korean Won)
- **Description**: Requested loan amount
- **Statistics**:
  - Mean: 413 (4,130,000 KRW)
  - Std Dev: 292.4
  - Median: 300 (3,000,000 KRW)
  - Range: 100–5,000 (1,000,000–50,000,000 KRW)
  - Q1: 200, Q3: 500

### 7. Loan Interest Rate

- **Type**: Continuous (ratio)
- **Unit**: Decimal (0–1 scale)
- **Description**: Annual interest rate for the loan
- **Statistics**:
  - Mean: 0.32 (32%)
  - Std Dev: 0.05
  - Median: 0.30 (30%)
  - Range: 0.05–0.59 (5%–59%)
  - Q1: 0.29, Q3: 0.35

### 8. Monthly DTI (Debt-to-Income Ratio)

- **Type**: Continuous (ratio)
- **Unit**: Decimal (unitless)
- **Description**: Monthly debt-to-income ratio
- **Calculation**: (Monthly debt payment) / (Monthly income)
- **Statistics**:
  - Mean: 0.37
  - Std Dev: 0.37
  - Median: 0.30
  - Range: 0.00–5.00
  - Q1: 0.15, Q3: 0.51
- **Note**: Values > 1.0 indicate debt payments exceed income

### 9. Loan Term

- **Type**: Discrete (integer)
- **Unit**: Months
- **Description**: Loan repayment period
- **Statistics**:
  - Mean: 19.6 months
  - Std Dev: 7.3 months
  - Median: 24 months
  - Range: 6–36 months
  - Q1: 12 months, Q3: 24 months
- **Common values**: 6, 12, 24, 36 months

### 10. Months of Service

- **Type**: Continuous (integer)
- **Unit**: Months
- **Description**: Number of months the borrower has been registered on the platform
- **Statistics**:
  - Mean: 29.0 months
  - Std Dev: 42.8 months
  - Median: 13 months
  - Range: 0–491 months
  - Q1: 4 months, Q3: 36 months
- **Note**: High variability indicates diverse user tenure

### 11. Number of Investors

- **Type**: Discrete (integer)
- **Unit**: Count
- **Description**: Number of individual investors who funded the loan
- **Statistics**:
  - Mean: 34.5 investors
  - Std Dev: 23.1 investors
  - Median: 30 investors
  - Range: 0–164 investors
  - Q1: 17 investors, Q3: 48 investors

## Text Variables (3 Fields)

### 1. Title

- **Type**: Free text (Korean)
- **Description**: Short title or headline of the loan application
- **Length statistics** (characters):
  - Mean: 15.7
  - Std Dev: 7.4
  - Median: 14.0
  - Range: 0–76
  - Q1: 10.0, Q3: 20.0
- **Example**: "긴급 생활자금 필요" (Urgent living expenses needed)

### 2. Loan Purpose

- **Type**: Free text (Korean)
- **Description**: Detailed explanation of why the borrower needs the loan
- **Length statistics** (characters):
  - Mean: 213.3
  - Std Dev: 217.4
  - Median: 152.0
  - Range: 0–2,433
  - Q1: 75.0, Q3: 272.0
- **Example**: "사업 운영자금 마련을 위해 대출이 필요합니다..." (Need loan for business operating capital...)

### 3. Repayment Plan

- **Type**: Free text (Korean)
- **Description**: Borrower's explanation of how they plan to repay the loan
- **Length statistics** (characters):
  - Mean: 225.6
  - Std Dev: 218.1
  - Median: 175.0
  - Range: 0–3,357
  - Q1: 83.0, Q3: 302.0
- **Example**: "매월 급여에서 일정 금액을 상환하겠습니다..." (Will repay a fixed amount from monthly salary...)

### Combined Text

- **Description**: Concatenation of Title + Loan Purpose + Repayment Plan
- **Length statistics** (characters):
  - Mean: 454.5
  - Std Dev: 350.9
  - Median: 370.0
  - Range: 7–3,946
  - Q1: 221.0, Q3: 585.0

## Target Variable

### Default Indicator

- **Type**: Binary categorical
- **Encoding**: 1 = Default, 0 = Repayment
- **Description**: Loan outcome (whether the borrower defaulted or repaid)
- **Distribution**:
  - Default (1): 3,352 cases (55.34%)
  - Repayment (0): 2,705 cases (44.66%)
- **Definition of default**: Failure to repay the loan according to the agreed terms

## Risk Groups (Derived Variables)

### High-Risk Group

- **Definition**: Bottom 30% of credit scores (≤ Q30)
- **Credit score threshold**: ≤ 596
- **Sample size**: 1,844 borrowers
- **Default rate**: 68.55%

### Low-Risk Group

- **Definition**: Top 30% of credit scores (≥ Q70)
- **Credit score threshold**: ≥ 691
- **Sample size**: 1,829 borrowers
- **Default rate**: 41.12%

## Text Length Groups (Derived Variables)

### Long Group

- **Definition**: Total text length ≥ median
- **Median threshold**: 370 characters
- **Description**: Borrowers who provided longer, more detailed narratives

### Short Group

- **Definition**: Total text length < median
- **Median threshold**: 370 characters
- **Description**: Borrowers who provided shorter, less detailed narratives

## Marginal vs. Clear Cases (Derived Variables)

### Marginal Cases

- **Definition**: Predicted default probability in [0.3, 0.7]
- **Description**: Cases where the structured-only model is uncertain
- **Sample size**: ~14% of test set (approximately 170 cases)

### Clear Cases

- **Definition**: Predicted default probability < 0.3 or > 0.7
- **Description**: Cases where the structured-only model is confident
- **Sample size**: ~86% of test set (approximately 1,040 cases)

## Data Preprocessing

### Structured Variables

1. **Scaling**: Continuous variables are standardized (mean=0, std=1) or normalized (min-max scaling)
2. **Binary variables**: Retain 0/1 encoding without scaling
3. **Missing values**: Handled as 0 (for credit score and income) or imputed with median

### Text Variables

1. **Concatenation**: Three text fields are concatenated into a single narrative
2. **Normalization**: Unicode normalization, lowercasing
3. **Tokenization**: Korean-specific tokenization (e.g., using KoNLPy)
4. **Stopword removal**: Common Korean stopwords removed
5. **Rare token filtering**: Tokens appearing in < 5 documents removed

## Train-Test Split

- **Split ratio**: 80% training, 20% test
- **Stratification**: Stratified by target variable to preserve class distribution
- **Random seed**: 42 (fixed for reproducibility)
- **Training set size**: 4,846 samples
- **Test set size**: 1,211 samples

## Data Quality Notes

1. **Missing values**: Some borrowers have credit score = 0 or income = 0, indicating missing data
2. **Outliers**: Some extreme values in Monthly DTI (> 1.0) and Months of Service (> 400)
3. **Text quality**: Text fields vary greatly in length and detail; some are very short or generic
4. **Class imbalance**: Default rate (55.34%) is slightly higher than repayment rate (44.66%)

## Data Ethics and Privacy

- **Anonymization**: All personally identifiable information (PII) has been removed
- **Aggregation**: Data is aggregated at the loan level, not individual level
- **Consent**: Data collection complied with platform terms of service and applicable privacy laws

## References

For more information on variable definitions and data collection procedures, see:

- Paper Section 2: Data
- Table 2-2: Structured Variables and Descriptive Statistics
- Table 2-4: Descriptive Statistics for Text Length

---

Last updated: 2026-02-01
