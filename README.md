# ProdigyInfoTech_TASK4
## TASK 4: Analyzing and Visualizing Sentiment Patterns in Social Media Data

This project analyzes sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands using the NLTK Vader sentiment analyzer.

## Dataset

The dataset used in this project is the Tweets dataset, containing tweets related to US airline sentiment.

## Steps

1. **Data Loading**: Load the dataset and explore its structure.
2. **Data Preprocessing**: Check for missing values and analyze sentiment distribution.
3. **Sentiment Analysis**: Use NLTK's Vader SentimentIntensityAnalyzer to calculate sentiment scores for each tweet.
4. **Visualization**: Visualize sentiment distribution, sentiment scores distribution, and sentiment scores by airline.

## How to Run

1. Clone the repository.
2. Ensure you have the necessary libraries installed (`pandas`, `matplotlib`, `seaborn`, `nltk`).
3. Place the dataset (`Tweets.csv`) in the same directory as the script.
4. Run the script (`sentiment_analysis.py`) to perform sentiment analysis and generate visualizations.

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv(r"C:\Users\91812\Pictures\Tweets.csv")

# Explore the structure and first few rows
display(df.head())

# Check for missing values
print(df.isnull().sum())
```
![Screenshot 2024-07-06 221930](https://github.com/PendemLikhitha/PRODIGY_DS_04/assets/159911587/2a064f66-5bd3-4967-861f-bb09c2d3c9ce)
![Screenshot 2024-07-06 221917](https://github.com/PendemLikhitha/PRODIGY_DS_04/assets/159911587/23bee56a-6060-4b58-bafb-d34732d40009)
![Screenshot 2024-07-06 223932](https://github.com/PendemLikhitha/PRODIGY_DS_04/assets/159911587/3d991343-7614-4004-9d0f-0f1ee25e66c6)


```python
# Define custom color palettes or colormaps
custom_palette = ["#8A9A5B", "#C9A9A6", "#FA5F55"]  # Example of custom colors

# Plot sentiment distribution with custom colors
plt.figure(figsize=(8, 6))
sns.countplot(x='airline_sentiment', data=df, palette=custom_palette)
plt.title('Sentiment Distribution in US Airline Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```
![Screenshot 2024-07-06 223941](https://github.com/PendemLikhitha/PRODIGY_DS_04/assets/159911587/7b91f5b3-f381-4ef5-a4f9-25529357a303)

```python
# Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each tweet
df['sentiment_scores'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df, x='sentiment_scores', bins=30, kde=True,color="#708090")
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.show()
```
![Screenshot 2024-07-06 223949](https://github.com/PendemLikhitha/PRODIGY_DS_04/assets/159911587/09f67afd-25f3-4269-99f8-709d32b69a03)
```python
# Visualize sentiment scores by airline
plt.figure(figsize=(12, 8))
sns.boxplot(x='airline', y='sentiment_scores', data=df, palette=custom_palette)
plt.title('Sentiment Scores by Airline')
plt.xlabel('Airline')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.savefig('assets/sentiment_scores_by_airline.png')  # Save the plot
plt.show()
```
![Screenshot 2024-07-06 223958](https://github.com/PendemLikhitha/PRODIGY_DS_04/assets/159911587/1b41d1e2-07f4-46fb-b4ff-6a9976ba60c7)

