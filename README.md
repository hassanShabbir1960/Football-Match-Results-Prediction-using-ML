<h1>Football Match Results Prediction using Machine Learning</h1>
<p>This project aims to predict the outcome of football matches in the English Premier League using machine learning. The project is divided into several parts, including web scraping, data cleaning and processing, feature engineering, and visualization. The following sections provide an overview of the project and how to run the code.</p>
<h2>Installation</h2>
<p>To run this project, you need to install the necessary libraries. You can do this by running the following command in your terminal:</p>
<pre><code>pip install -r requirements.txt</code></pre>
<h2>1. Data Scraping</h2>
<p>In this section, we scrape match and shooting data for English Premier League teams over a range of years. We use the Requests library to fetch webpages, BeautifulSoup to parse HTML content, and Pandas to handle and manipulate data in DataFrames. We start by importing necessary libraries and defining relevant functions to simplify the main process. We then iterate over the desired years, retrieve the league table, and extract the links to individual team pages. Next, we iterate through each team's URL, retrieve match data and shooting data, and merge them into a single DataFrame based on the match date. The final result is a DataFrame containing match information and shooting statistics for each team's matches in the Premier League over the specified years.</p>

<pre><code>python scrapper.py</code></pre>

<h2>2. Data Cleaning and Processing</h2>
<p>In this code, we are performing data cleaning and processing on our scraped data. We start by identifying and removing columns that have the same value for all rows or consist only of NaN values. Once we have this information, we remove these columns from the DataFrame.</p>
<p>Next, we check for missing values in the DataFrame. By examining the output, we can determine which columns have missing values and decide on further processing steps. This allows us to identify any issues in the data and handle them accordingly, ensuring that the data is clean and suitable for further analysis.</p>
<pre><code>python DataPreprocessor.py</code></pre>

<h2>3. Visualization</h2>
<p>In this code, we create visualizations of the data using Plotly. We create a heatmap plot to display the goals scored by each team over time, a scatter plot to display the relationship between expected goals and actual goals for each team in the English Premier League, and calculate the average possession by team and sort by possession. We also calculate the number of goals scored and passes made by each player and group the data by team and venue to count the number of wins.</p>

<pre><code>python DataVisualizer.py</code></pre>

<h2>4. Feature Engineering</h2>
<p>In this code, we perform feature engineering on the data by dropping irrelevant columns, creating rolling averages for specified columns in a group, creating a new column containing codes for the days of the week in a specified date column, creating a new column containing codes for the unique values in a specified opponent column, and creating a new column containing codes for the unique values in a specified venue column.</p>
<pre><code>python FeatureEngineer.py</code></pre>


<h2>5. Model Training and evaluation</h2>
<p>In this code, we train and evaluate five different models using the preprocessed and feature engineered data. We use the following models: Logistic Regression, Random Forest Classifier, Support Vector Machine Classifier, Gradient Boosting Classifier, and Multinomial Naive Bayes Classifier. For each model, we perform 10-fold cross-validation and calculate the mean and maximum accuracy. We also visualize the confusion matrix for the iteration with the highest accuracy. The code can be run using the following command:</p>
<pre><code>python train.py</code></pre>

## Comparison of Models Performance

| Model        | Mean Accuracy | Best Accuracy |
| ------------ | -------------| ------------- |
| Random Forest| 0.86         | 0.91          |
| SVM          | 0.73         | 0.75          |
| Logistic Reg.| 0.72         | 0.73          |
| Naive Bayes  | 0.68         | 0.71          |
| Decision Tree| 0.65         | 0.68          |
