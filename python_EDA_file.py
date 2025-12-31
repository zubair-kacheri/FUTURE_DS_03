import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("E:\python\Future interns\Student_Satisfaction_Survey.csv",encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip()

# print(df.head())

weight_cols = ['Weightage 1','Weightage 2','Weightage 3','Weightage 4','Weightage 5']

df['Total_Responses'] = df[weight_cols].sum(axis=1)

df['Avg_Rating'] = (
    df['Weightage 1']*1 +
    df['Weightage 2']*2 +
    df['Weightage 3']*3 +
    df['Weightage 4']*4 +
    df['Weightage 5']*5
) / df['Total_Responses']

def label_satisfaction(x):
    if x >= 4:
        return 'Highly Satisfied'
    elif x >= 3:
        return 'Satisfied'
    else:
        return 'Needs Improvement'

df['Satisfaction_Level'] = df['Avg_Rating'].apply(label_satisfaction)


df['Sentiment'] = pd.cut(
    df['Avg_Rating'],
    bins=[0, 3, 4, 5],
    labels=['Negative', 'Neutral', 'Positive'],
    include_lowest=True
)

# ------------------------------------------------------------------------------------------
# Average Satisfaction Rating by Course

plt.figure(figsize=(10,5))
sns.barplot(data=df, x='Course Name', y='Avg_Rating')
plt.xticks(rotation=90)
plt.title("Average Satisfaction Rating by Course")
plt.show()

# ----------------------------
# Distribution of Total Feedback Given

plt.figure(figsize=(7, 5))
plt.hist(df['Total Feedback Given'], bins=20)
plt.title("Distribution of Total Feedback Given")
plt.xlabel("Total Feedback Given")
plt.ylabel("Count")
# plt.show()

# ----------------------------------
# "Distribution of Total Configured"
plt.figure(figsize=(7,5))
sns.histplot(df['Total Configured'], bins=20)
plt.title("Distribution of Total Configured")
plt.xlabel("Total Configured")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# "Total Feedback Given vs Total Configured"

plt.figure(figsize=(7,5))
sns.scatterplot(x='Total Configured', y='Total Feedback Given', data=df)
plt.title("Total Feedback Given vs Total Configured")
plt.xlabel("Total Configured")
plt.ylabel("Total Feedback Given")
plt.show()


# ----------------------------------------------------

plt.figure(figsize=(7,5))
sns.histplot(df['Average/ Percentage'], bins=15)
plt.title("Distribution of Weighted Average Ratings")
plt.xlabel("Weighted Avg Rating")
plt.ylabel("Count")
plt.show()

# -------------------------------------------------------
df['Satisfaction_Level'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(5,5))
plt.title("Overall Student Satisfaction Level")
plt.ylabel('')
plt.show()

# --------------------------------------------------------
sentiment_counts = df['Sentiment'].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title("Sentiment Analysis Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Records")
plt.show()