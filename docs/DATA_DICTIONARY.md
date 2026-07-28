# Data Dictionary

Raw file: `data/raw/sentiment_scoring.25.12.30.xlsx` (restricted; 6,057 rows).

## Target

| Column (KR) | Definition |
|---|---|
| 상환결과 | Repayment outcome; 채무불이행 (default) → 1, otherwise → 0 |

## Structured variables (13) — paper Table 2

| Column (KR) | Paper name | Notes |
|---|---|---|
| 신청금액(만원) | Loan Amount | 10,000 KRW units |
| 신청금리 | Interest Rate | |
| 대출용도(대출상환0) | Loan Purpose | 0 = debt consolidation, 1 = other |
| 대출시기 | Loan Timing | application year (7–15 ↔ 2007–2015) |
| 투자인원 | Number of Investors | |
| 나이 | Age | |
| 신용평점 | Credit Score | 0–950 |
| 근무개월 | Employment Months | |
| 4대보험(가입0) | Insurance Status | 0 = enrolled, 1 = not enrolled |
| 대출(은행보험) | Bank Loan | outstanding bank/insurance loan, 10,000 KRW |
| 총횟수 | Total Applications | prior applications |
| 성공횟수 | Successful Applications | |
| 성공률 | Success Rate | 0–1 |

Continuous variables are standardized within training folds; binary variables
keep 0/1 coding.

## Text fields (3)

| Column (KR) | Paper name |
|---|---|
| 제목 | Title |
| 신청목적 | Loan purpose (narrative) |
| 상환계획 | Repayment plan (narrative) |

The three fields are concatenated into a single narrative per application.
