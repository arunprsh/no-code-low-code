| Column name       | Description     | 
| :------------- | :---------- | 
|`loan_status`|`Current status of the loan (target variable)`.|
|`loan_amount`|The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.| 
|funded_amount_by_investors| The total amount committed by investors for that loan at that point in time.| 
|term|The number of payments on the loan. Values are in months and can be either 36 or 60.|
|interest_rate|Interest Rate on the loan|
|installment|The monthly payment owed by the borrower if the loan originates.|
|grade|LC assigned loan grade|
|sub_grade|LC assigned loan subgrade|
|employment_length|Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.|
|home_ownership|The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.|
|annual_income|The self-reported annual income provided by the borrower during registration.|
|verification_status|Indicates if income was verified by LC, not verified, or if the income source was verified|
|issued_amount||
|purpose|A category provided by the borrower for the loan request.|
|dti|A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.|
|earliest_credit_line|The month the borrower's earliest reported credit line was opened|
|inquiries_last_6_months|The number of inquiries in past 6 months (excluding auto and mortgage inquiries)|
|open_credit_lines|The number of open credit lines in the borrower's credit file.|
|derogatory_public_records|Number of derogatory public records|
|revolving_line_utilization_rate|Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.|
|total_credit_lines|The total number of credit lines currently in the borrower's credit file|