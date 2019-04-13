import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
project_titles = pd.read_csv('Asset_Id_Titles')


def visualize(df):
  sns.set_style('white')
  ratings =pd.DataFrame(df.groupby('title')['rating'].mean())
  ratings['rating_numbers'] = pd.DataFrame(df.groupby('title')['rating'].count())
  ratings['rating_numbers'].hist(bins=70)
  ratings['rating'].hist(bins=70)
  sns.jointplot(x='rating', y='rating_numbers', data=ratings, alpha=0.5)

def recommendation(projectname):
  projectmat = df.pivot_table(index='user_id', columns='title', values='rating')
  ratings.sort_values('rating_numbers', ascending=False)
  similar_to_projectname= projectmat.corrwith(projectname)
  corr_acc = pd.DataFrame(similar_to_projectname, columns=['Correlation'])
