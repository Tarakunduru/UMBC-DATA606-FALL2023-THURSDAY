##  Title and Author
    
-	Credit card fraud detection
*	Prepared for UMBC Data Science master’s degree Capstone by Dr Chaojie (Jay) Wang
* Sai Tara Kunduru
* [GitHub] https://github.com/Tarakunduru
* [LinkedIn] www.linkedin.com/in/tarakunduru
* [Presentation File] (Will Update later)
* [YouTube Video] (Will update the link later)

##  Background

## What is it about? 
Credit card fraud detection is a critical application of machine learning in the financial sector. It involves using computational techniques to identify and prevent unauthorized or fraudulent credit card transactions. This process is essential for safeguarding the financial interests of both cardholders and financial institutions. Here are the key aspects of credit card fraud detection using machine learning:
	
## Why does it matter? 
Credit card fraud detection is crucial for safeguarding individuals and financial institutions from monetary losses. Without effective detection, fraudulent transactions can result in significant financial harm, impacting both cardholders and banks. Detecting fraud allows for timely intervention, reducing financial losses for all parties involved. Credit card transactions involve sensitive personal and financial information. Detecting fraud helps protect this data from being exploited by malicious actors. By keeping data secure, credit card fraud detection not only prevents financial losses but also reduces the risk of identity theft and other forms of cybercrime.

## What are your research questions? 
Are there time-related trends in credit card transactions, and can we predict future transaction behavior?
Can we identify distinct customer segments based on transaction behavior, and how can this information be used for targeted marketing or risk assessment?
How can machine learning models effectively detect fraudulent credit card transactions using the provided features?

## DATA 
* Data Source: https://www.kaggle.com/datasets/dermisfit/fraud-transactions-dataset
* Data size: 351.2 mb
* Rows and columns:1048576*23
* What does each row represents: fraud

## Data Dictionary:


- trans_date_trans_time: It is a date format. It represents the date and time of the transaction.
- cc_num: it is an integer and represents the credit card number.
- merchant: It is a string that represents the merchant who was getting paid.
- category: It is a string that represents in what area does that merchant deal.
- amt: It is a float that represents the amount of money in American Dollars.
- first: It is a string that represents the first name of the card holder.
- last: It is a string that represents the last name of the card holder.
- gender: It is a category that represents the gender of the cardholder that is male and female!
- street:It is a string that represents the street of card holder residence
- city:It is a string that represents the city of card holder residence
- state:It is a string that represents the state of card holder residence
- zip:It is a integer that represents the ZIP code of card holder residence
- lat:It is a float that represents the latitude of card holder
- long:It is a float that represents the longitude of card holder
- city_pop:It is a integer that represents the population of the city
- job:It is a string that represents the trade of the card holder
- dob:It is a date that represents the date of birth of the card holder
- trans_num: It is a string that represents the Transaction ID
- unix_time: It is an integer that represents the Unix time which is the time calculated since 1970 to today.
- merch_lat: It is a float that represents latitude of the merchant
- merch_long:It is a float that represents longitude of the merchant
- is_fraud: It is a categorical value that represents Whether the transaction is fraud(1) or not(0)








```python

```


