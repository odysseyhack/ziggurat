
# coding: utf-8

# ### Import the libraries

# In[2]:


import numpy as np
import pandas as pd


# ### Get the data

# In[3]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']


# In[4]:


df = pd.read_csv('u.data', sep='\t', names=column_names)


# In[5]:


df.head()


# In[12]:


project_titles = pd.read_csv('Asset_Id_Titles')


# In[13]:


project_titles.head()


# Merge them together:

# ### Import vizualisation libraries

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


sns.set_style('white')


# In[20]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)


# In[21]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)


# ##### create a ratings dataframe with average rating and number of ratings:

# In[22]:


ratings =pd.DataFrame(df.groupby('title')['rating'].mean())


# In[23]:


ratings.head()


# #####  Set the number of ratings column:

# In[24]:


ratings['rating_numbers'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[25]:


ratings.head()


# ##### Number of ratings histogram

# In[26]:


ratings['rating_numbers'].hist(bins=70)


# #### Average rating per movie histogram

# In[27]:


ratings['rating'].hist(bins=70)


# ##### Relationship between the average rating and the actual number of ratings
# ###### The larger the number of ratings, the project likely the rating of a project is

# In[64]:


sns.jointplot(x='rating', y='rating_numbers', data=ratings, alpha=0.5)


# ## Recommending project
# 

# Let's create a matrix that has the user ids on one access and the project title on another axis. Each cell will then consist of the rating the user gave to that project.

# In[28]:


projectmat = df.pivot_table(index='user_id', columns='title', values='rating')
projectmat.head()


# ##### Most rated projects

# In[29]:


ratings.sort_values('rating_numbers', ascending=False).head(10)


# #### Let's choose two projects for our system: ACC, And MFSL

# What are the user ratings for those two projects?

# In[36]:


ACC_user_ratings = projectmat['ACC']
MFSL_user_ratings =projectmat['MFSL']


# In[37]:


ACC_user_ratings.head()


# #### correlation of every other project to that specific user behaviour on the ACC project

# In[38]:


similar_to_acc = projectmat.corrwith(ACC_user_ratings)
similar_to_acc.head()


# #### correlation of every other project to that specific user behaviour on the MFSL project

# In[39]:


similar_to_MFSL = projectmat.corrwith(MFSL_user_ratings)
similar_to_MFSL.head()


# ##### remove the NaN values and use a DF instead of Series

# In[40]:


corr_acc = pd.DataFrame(similar_to_acc, columns=['Correlation'])
corr_acc.dropna(inplace=True)


# In[41]:


corr_acc.head()


# ### Perfectly correlated project with ACC? 
# 

# In[42]:


corr_acc.sort_values('Correlation', ascending=False).head(10)


# #### Set a threshold for the number of ratings necessary and filter out project that have less than a certain number of rates

# join the 'number of ratings' column to our dataframe

# In[43]:


corr_acc = corr_acc.join(ratings['rating_numbers'], how='left', lsuffix='_left', rsuffix='_right')
corr_acc.head()


# filter out projects that have less than 100 rates (this value was chosen based off the ratings histogram from earlier)

# In[44]:


corr_acc[corr_acc['rating_numbers']>100].sort_values('Correlation', ascending=False).head()


# ### Perfectly correlated projects with MFSL?

# In[45]:


corr_MFSL = pd.DataFrame(similar_to_MFSL, columns=['Correlation'])
corr_MFSL.head()


# ##### remove the NaN values and use a DF instead of Series

# In[46]:


corr_MFSL.dropna(inplace=True)


# In[47]:


corr_MFSL = corr_MFSL.join(ratings['rating_numbers'], how='left')
corr_MFSL.head()


# filter out movies that have less than 100 reviews (this value was chosen randomly)

# In[48]:


corr_MFSL[corr_MFSL['rating_numbers']>100].sort_values('Correlation', ascending=False).head()

