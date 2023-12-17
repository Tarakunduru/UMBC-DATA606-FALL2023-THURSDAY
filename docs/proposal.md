# Hotel Booking Cancellation Prediction

* Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang
* Author: Sai Tara Kunduru
* GitHub : https://github.com/Tarakunduru
* Linkedin : https://www.linkedin.com/in/tarakunduru

## 2. Background

### What is it about?

The Hotel Booking Cancellation Prediction project leverages historical booking data to predict whether a hotel booking will be canceled. the dataset contains various features related to bookings. The primary objective is to develop a machine learning model for accurate predictions to assist hotel managers in inventory management and pricing strategies. The project will involve data preprocessing, EDA, feature engineering, model training, and deployment. The project aims to improve hotel revenue, reduce operational costs, and enhance customer satisfaction despite the challenges of imbalanced data, missing values, and model integration.

### Why does it matter?

The Hotel Booking Cancellation Prediction project matters because it helps hotels optimize revenue, manage room inventory, improve customer service, and make informed strategic decisions. By accurately predicting cancellations, hotels can adjust pricing, accommodate guests during peak periods, enhance customer satisfaction, and gain a competitive advantage. This project is crucial for reducing financial impact and improving overall business outcomes in the highly competitive hospitality industry.

### What are your reasearch questions?

What features in the booking data are the most significant predictors of booking cancellations?

How does the performance of the cancellation prediction model vary across different hotels or hotel chains?

Can the model's predictions be used to optimize pricing strategies and improve revenue management?


## 3. Data

* Data Source: This dataset is obtained from kaggle and is related to Hotel Booking Cancellation Prediction
* Data shape: Number of rows = 119210
              Number of columns = 32

* Data Dictionary
1. hotel: Type of hotel (e.g., City hotel, Resort hotel).
2. is_canceled: Binary value indicating if the booking was canceled (1) or not (0).
3. lead_time: Number of days between booking and arrival date.
4. arrival_date_year: Year of arrival date.
5. arrival_date_month: Month of arrival date.
6. arrival_date_week_number: Week number of arrival date.
7. arrival_date_day_of_month: Day of arrival date.
8. stays_in_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed.
9. stays_in_week_nights: Number of weeknights (Monday to Friday) the guest stayed.
10. adults: Number of adults in the booking.
11. children: Number of children in the booking.
12. babies: Number of babies in the booking.
13. meal: Type of meal package (e.g., Bed & Breakfast, Half board).
14. country: Country of origin of the booking.
15. market_segment: Market segment (e.g., Online TA, Offline TA, Direct).
16. distribution_channel: Booking distribution channel (e.g., TA/TO, Direct).
17. is_repeated_guest: Binary value indicating if the guest is a repeat customer (1) or not (0).
18. previous_cancellations: Number of previous bookings that were canceled by the guest.
19. previous_bookings_not_canceled: Number of previous bookings that were not canceled by the guest.
20. reserved_room_type: Type of room reserved.
21. assigned_room_type: Type of room assigned to the guest.
22. booking_changes: Number of times the guest made changes to the booking.
23. deposit_type: Type of deposit made (e.g., No Deposit, Non-refundable).
24. agent: ID of the travel agency that made the booking.
25. company: ID of the company that made the booking.
26. days_in_waiting_list: Number of days the booking was on the waiting list before confirmation.
27. customer_type: Type of customer (e.g., Transient, Contract).
28. adr: Average daily rate.
29. required_car_parking_spaces: Number of required car parking spaces.
30. total_of_special_requests: Number of special requests made by the guest.
31. reservation_status: Current status of the reservation (e.g., Check-Out, Canceled).
32. reservation_status_date: Date when the last status change occurred.

* Which variable/column will be your target/label in your ML model?

In the Hotel Booking Cancellation Prediction project, the target variable (also known as the label) will be the is_canceled column. This binary variable indicates whether a booking was canceled (1) or not (0). The goal of the machine learning model will be to predict this variable based on the other features in the dataset.

# 4. Exploratory Data Analysis

### 4.1.1 Checking and removing duplicates from the Data Set

* Checking the duplicates from the data set 

### 4.1.2 Analysing the data and visualization

* This plot is designed to show the relationship between two variables: whether a hotel booking was canceled (is_canceled) and whether the guest who made the booking is a repeated guest (is_repeated_guest).
* The is_canceled column likely contains binary values (0 or 1) indicating whether each booking was canceled or not, and the is_repeated_guest column also likely contains binary values indicating whether each guest is a repeated guest or not.
* The bars in the plot represent the frequency of each combination of is_canceled and is_repeated_guest values in the data.
  ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/de9731e8-7e3c-435e-89b6-e596390c60d1)

* this plot visualize the distribution of total special requests (total_of_special_requests) made by guests in the hotel_data DataFrame. 
* The x-axis of the histogram represents the different counts of special requests, and the y-axis represents the frequency of each count. 
* The seaborn function histplot is used to create the histogram, with x set to 'total_of_special_requests' 
* The title "distribution of total special requests" is set using plt.title(). This plot is intended to provide a visual representation of how frequently different numbers of special requests are made by guests.
  ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/ad025964-3903-4efa-a62b-cb37099cbfbe)

* This plot explore the relationship between reserved room types (reserved_room_type), lead time for booking (lead_time), and hotel type (hotel) in the hotel_data DataFrame.
* The x-axis represents the different room types, while the y-axis represents the lead time for bookings. The catplot function is used with data set to hotel_data, x set to 'reserved_room_type', y set to 'lead_time', and col set to 'hotel'.
* This creates separate plots for each hotel type. The color of the plot points is set to green using the color parameter.
   ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/baae8d8d-8a52-4e1b-b777-b1f22ba265d8)


* This plot generates to visualize the relationship between booking cancellations (is_canceled), the number of days in the waiting list (days_in_waiting_list), and the type of deposit (deposit_type). 
* In the plot, the x-axis represents booking cancellations, the y-axis represents the number of days in the waiting list, and the col parameter is used to create separate plots for each deposit type. 
* y set to 'days_in_waiting_list', x set to 'is_canceled', and col set to 'deposit_type'. All the data points are colored red, as specified by the color parameter.
  ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/1fd9bdbf-b3c3-414a-b3c7-6751e5665771)


* This plot displays the distribution of hotel bookings across different market segments (market_segment) and their cancellation status (is_canceled).
* In the plot, the x-axis represents the market segments, and the y-axis represents the count of bookings. 
* The plot is further enhanced by rotating the x-axis labels by 65 degrees and aligning them to the right for better readability, as specified by the set_xticklabels method.
 ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/9510e339-512a-4682-a290-88c2b1240126)

# 5. Feature selection methods

## 5.1 Univariate feature selection
* Features in matrix X are scaled using Min-Max scaling with MinMaxScaler().
* Utilizing SelectKBest with chi-squared (chi2) scoring, the code conducts independent feature selection based on significance to the target variable y.
* Feature scores and names are stored in dfscores and dfcolumns DataFrames, concatenated into featureScores for streamlined analysis.
* The nlargest() method extracts and prints the top 10 features by chi-squared scores, providing a concise summary.
* The printed DataFrame with 'columns' and 'Score' aids in identifying influential variables for subsequent modeling. Univariate selection simplifies models and boosts predictive performance by focusing on key features.

   ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/14b1f6e7-e4d6-4f46-b1d5-19fddeff1d97)

## 5.2 Correlation matrix with heatmap
* The image computes a correlation matrix (data_corr) for the hotel_data, revealing relationships between variables.
* A 12x9-inch heatmap is created using seaborn (sns.heatmap()) with the "RdYlGn" color map, visually representing the correlations.
* The resulting heatmap is displayed using plt.show(), offering an intuitive overview of the dataset's correlation structure.

  ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/d65e9280-1f06-4cc0-920c-f5b2b7a4b0dd)

## 5.3 Top Features
* The image defines top_n as the number of features to display and extracts the top N important features using nlargest() on feat_importances.
* A horizontal bar graph is created using top_features.plot(kind='barh') for visualizing the importance scores of the selected features.
* The resulting bar graph is shown with plt.show(), offering a concise visual summary of the top features.

  ![image](https://github.com/Tarakunduru/UMBC-DATA606-FALL2023-THURSDAY/assets/143665432/71834da8-5e2b-4043-bbd7-076331a2d423)







## Conclusions:

Cancellation count is less for repeated guests

Distribution of total special requests is left skewed

Lead time is high for room types C,A and D for both the hotels

Deposit type is refundable waiting list days are same but the other categories it is high.

## References:

* M. R. H. Subho, M. R. Chowdhury, D. Chaki, S. Islam and M. M. Rahman, "A Univariate Feature Selection Approach for Finding Key Factors of Restaurant Business," 2019 IEEE Region 10 Symposium (TENSYMP), Kolkata, India, 2019, pp. 605-610, doi: 10.1109/TENSYMP46218.2019.8971127.
* V. Aggarwal, V. Gupta, P. Singh, K. Sharma and N. Sharma, "Detection of Spatial Outlier by Using Improved Z-Score Test," 2019 3rd International Conference on Trends in Electronics and Informatics (ICOEI), Tirunelveli, India, 2019, pp. 788-790, doi: 10.1109/ICOEI.2019.8862582.
* G. König, C. Molnar, B. Bischl and M. Grosse-Wentrup, "Relative Feature Importance," 2020 25th International Conference on Pattern Recognition (ICPR), Milan, Italy, 2021, pp. 9318-9325, doi: 10.1109/ICPR48806.2021.9413090.

