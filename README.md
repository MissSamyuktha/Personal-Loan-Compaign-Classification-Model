# Personal-Loan-Compaign-Classification-Model
Built a model using sklearn decision tree to predict whether a liability customer will buy personal loans, to understand which customer attributes are most significant in driving purchases, and to identify which segment of customers to target more that will help the marketing department to identify the potential customers.

## **Context**
AllLife Bank is a US bank that has a growing customer base. The majority of these customers are liability customers (depositors) with varying sizes of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio. You as a Data Scientist at AllLife Bank have to build a model that will help the marketing department to identify the potential customers who have a higher probability of purchasing the loan.

## **Objective**
To predict whether a liability customer will buy personal loans, to understand which customer attributes are most significant in driving purchases, and to identify which segment of customers to target more

## **Data Dictionary**
- ID: Customer ID
- Age: Customer’s age in completed years
- Experience: # years of professional experience
- Income: Annual income of the customer (in thousand dollars)
- ZIP Code: Home Address ZIP code.
- Family: The family size of the customer
- CCAvg: Average spending on credit cards per month (in thousand dollars)
- Education: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
- Mortgage: Value of house mortgage if any. (in thousand dollars)
- Personal_Loan: Did this customer accept the personal loan offered in the last campaign?
- Securities_Account: Does the customer have a securities account with the bank?
- CD_Account: Does the customer have a certificate of deposit (CD) account with the bank?
- Online: Do customers use Internet banking facilities?
- CreditCard: Does the customer use a credit card issued by any other Bank (excluding All Life Bank)?

## Tech Stack Used:
- **Pandas & Numpy:** for data manipulation
- **Matplotlib & Seaborn:** for data visualization
- **Scikit-learn, Python:** for machine learning models
- **StandardScaler, Log transformations, train-test-split:** for data preparation for modelling
- **Decision Trees:** for model building
- **Precision, Recall, Accuracy, F1 Score:** metrics for model evaluation

## Project:
This project is part of **Post Graduation Program in AI & ML** at *UT Austin* delivered via *Great Learning*.   
[Project link:] (https://www.mygreatlearning.com/eportfolio/samyuktha25)

## Data Insights, Decision Tree Models & Conclusions
**Data Overview**
- The Dataset has 5000 rows and 14 columns. Data in each row corresponds to each customer attributes stored in AllLife Bank customer base.

- All 14 columns contains numeric data type(int and float). Total memory usage is approximately 547.0 KB.

- Education, Personal_Loan, Securities_Account, CD_Account, Online, CreditCard, although interpreted here as numerical, are categorical variables that are encoded by default.

- No null values values present. No data is missing. No duplicates found.

**Brief Statistical Summary of Data**
- Average age of customers is 45 year old with a mean experience of 20 years.
- Average income is around 73,000 dollars.
- 50% of them are either single or a family of 2.
- Amount spent on credit card per month on an average is ~1938 dollars, with more than 50% customers spending below average.
- Half of them not on any mortgage
- 9% of them took a personal loan post compaign.
- 6% hold CD account with the bank.
- Around 60% of customers use online banking services.
- Around 20% of them hold credit cards issued by other banks.

 **Exploratory Data Analysis**, **Log transformations**, **StandardScaler** done on data and is split for training and test data as part of **data preparation**.

**Model building** using default sklearn, with class weights, pruned models in order to reduce model complexity, experimenting with threshold values for better classification of model.

**Model evaluation** using model performance metrics - Accuracy, Recall, Precision, F1 Score.

**Key Reasons for Prioritizing Recall**
- Avoid Losing Potential Customers: A false negative means a customer who actually wants a loan is misclassified as uninterested, causing lost revenue.
- Marketing Cost Efficiency: High recall ensures the bank targets more actual loan acceptors, optimizing advertising efforts.
- Balancing Business Goals: While precision matters (to reduce false positives), recall ensures as many interested customers as possible are identified.

**Trade-Off Between Recall & Precision**
- High Recall: Captures almost all loan acceptors, but may lead to some false positives (non-loan takers included in marketing).

- High Precision: Ensures most predicted loan acceptors are correct, but misses some actual acceptors (lower recall).tradeoffs

**Best Model**

**Pre-pruning with threshold 0.6** achieved the best balance.
- It optimized precision (0.79) while maintaining high recall (0.95), ensuring targeted outreach to real loan acceptors while reducing false positives.

**Post-pruning** improved recall but reduced precision. With ccp_alpha = 0.00215, recall reached 1.0 but precision dropped to 0.74, suggesting slightly weaker selectivity in identifying true loan takers.

**Feature Importance**
- Pre-Pruned Model: log_Income, log_CCAvg, Education, Family were key.

- Post-Pruned Model: log_Income dominates even more, family retains strong influence, Education lowered while log_CCAvg influence reduced.

- Income remains the strongest feature.

**Threshold tuning** significantly impacted precision: Raising the threshold from **0.5 to 0.6** helped reduce false positives across models.

## Key Takeaways for the Marketing Team
After an in-depth Exploratory Data Analysis (EDA) and model evaluation, here are the critical insights for refining loan marketing strategies:

**1. High-Income Earners Are More Likely to Accept Loans**
- Customers with higher income levels exhibit a strong correlation with loan acceptance.
- Marketing should prioritize premium loan products tailored to high-income earners.
- Actionable Strategy: Offer flexible loan structures and investment-linked financing for affluent customers.

**2. Education Plays a Key Role in Loan Acceptance**
- Advanced degree holders are most likely to accept loans due to higher salaries and financial stability.
- Undergraduates have the lowest acceptance rate, indicating possible income constraints.
- Actionable Strategy:
- - Premium loan offers for advanced degree holders.
- - Starter loan packages & financial literacy programs for undergraduates to introduce credit-building opportunities.

**3. Credit Card Usage Impacts Borrowing Behavior**
- Higher credit card spending (CCAvg) correlates with loan acceptance.
- Low-spenders show hesitancy toward personal loans.
- Actionable Strategy:
- - Segment customers based on spending patterns to personalize loan offerings.
- - Encourage low-spenders with targeted financial products & lower-risk loan options.

**4. Mortgage Holders Show Varied Loan Behavior**
- Non-mortgage holders show stronger personal loan acceptance trends.
- Investigate high mortgage outliers: Some high-mortgage customers reject personal loans, possibly due to pre-existing financial commitments.
- Actionable Strategy:
Refine loan marketing for mortgage holders, focusing on refinancing or bundled financial products.
- Prioritize personal loans for non-mortgage holders, aligning with liquidity needs.


**5. Family Size Influences Loan Demand**
- Larger families (3–4 members) show higher loan acceptance, possibly due to financial responsibilities.
- Single-member families tend to reject loans, likely due to financial independence.
- Actionable Strategy:
- - Introduce family-focused loan products, emphasizing flexible repayment plans.
- - Personalize offers for single-member families, emphasizing investment-based financial solutions.

**6. Securities & CD Accounts Affect Loan Preferences**
- Customers with securities accounts or CDs are more likely to take loans.
- Actionable Strategy:
- - Target CD holders with personal loan promotions.
- - Explore cross-selling strategies for securities account holders, integrating investment-linked loan products.

**7. Optimized Customer Segmentation for Better Targeting**
- Use A/B Testing for Campaign Effectiveness: Run tests on different marketing groups and loan terms to maximize impact
- Decision tree-based feature importance analysis highlights top contributors to loan acceptance.
- Actionable Strategy:
- - Deploy precision-targeted campaigns using machine learning-driven customer segmentation.
- - Optimize marketing outreach by integrating behavior-based financial profiling.

## **Next Steps for the Marketing Team**

- Use **Pre-Pruned Model with Threshold = 0.6** for Deployment. This gives the best mix of precision & recall, ensuring efficient marketing with minimal wasted efforts.
- Refine marketing strategies by segmenting customers based on financial behavior insights.
- Tailor personalized loan offers by combining credit card usage, education, and income levels.
- Introduce specialized loan packages for unique customer groups, including families, high-income individuals, and investment-focused customers.
- Leverage decision tree models built to improve campaign efficiency and customer targeting.  
This strategy ensures higher loan adoption rates, stronger customer engagement, and optimized financial products for diverse customer needs.




