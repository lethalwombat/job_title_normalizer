#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[ ]:


import pandas as pd


# Import data

# In[ ]:


# job titles from rocket
df_test_data = (
    pd.read_csv('job_titles_rocket.gz', delimiter='\t', encoding = "ISO-8859-1", names=['job_title_raw', 'n'])
#     pd.read_csv('job_titles_rocket.gz', delimiter='\t', names=['job_title', 'n'])    
    .dropna()
    .reset_index(drop=True)
)

# transform the data
df_test_data = (
    df_test_data
    .assign(
        job_title_raw=df_test_data['job_title_raw'].str.lower().str.strip()
    )
)

# import assigned roles from thesaurus
df_assigned_roles = (
    pd.read_csv('job_title_dictionary.txt', delimiter='\t')
    .query('FindPhraseStatus == "assignedrole"')
    .drop(columns=['FindPhraseStatus'])
    .reset_index(drop=True)
)

# transform role data
df_assigned_roles = (
    df_assigned_roles
    .assign(
        job_title_raw=df_assigned_roles['FindPhrase'].str.lower().str.strip(),
        job_title=df_assigned_roles['ReplacePhrase'].str.lower().str.strip()
    )
    [['job_title_raw', 'job_title']]
)


# Find out exact matches

# In[ ]:


# first merge
df_exact_matches = (
    df_test_data
    .merge(df_assigned_roles, how='outer', left_on='job_title_raw', right_on='job_title_raw', indicator=True)
)

# split data into matches and non-matches
df_exact_non_matches = (
    df_exact_matches
    .query('_merge == "left_only"')
    .drop(columns=['_merge', 'job_title'])
    .reset_index(drop=True)
)
df_exact_matches = (
    df_exact_matches
    .query('_merge == "both"')
    .drop(columns=['_merge'])
    .reset_index(drop=True)    
)

# display stats
print('Total records: {}'.format(df_test_data.shape[0]))
print('Matched: {}'.format(df_exact_matches.shape[0]))
print('Unmatched: {}'.format(df_exact_non_matches.shape[0]))


# Define matching function

# In[ ]:


# %%time
from rapidfuzz import process
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=4)

def match_job_titles(df, dictionary=None, score_cutoff=90):
    # nothing to look up
    if dictionary is None:
        print('No dictionary set!')
        return df
    
    # get roles to look up 
    _roles = dictionary['job_title_raw'].unique()

    # matching function
    _match_func = lambda x : process.extractOne(x, _roles)

    # apply matching function
    df = (
        df
        .assign(
            _match=df['job_title_raw'].parallel_apply(_match_func)
        )
    )
    # filter out bad quality matches
    df = (
        df
        .assign(
            job_title_matched=df['_match'].apply(lambda x : x[0]),            
            _score=df['_match'].apply(lambda x : x[1]),
        )
        .query('_score >= @score_cutoff')
        .drop(columns=['_match', '_score'])
        .reset_index(drop=True)
    )
    return df

# apply fuzzy matching and match back to the dictionary
df_fuzzy_matches = (
    df_exact_non_matches
#     .head(10000) # testing purposes
    .pipe(match_job_titles, df_assigned_roles)
    [['job_title_matched', 'n']]
    .merge(df_assigned_roles, how='left', left_on='job_title_matched', right_on='job_title_raw')
    .drop(columns=['job_title_matched'])
    [['job_title_raw', 'n', 'job_title']]    
)

# display stats
print('Total records: {}'.format(df_test_data.shape[0]))
print('Exact matches: {}'.format(df_exact_matches.shape[0]))
print('Fuzzy matches: {}'.format(df_fuzzy_matches.shape[0]))


# Combine results

# In[ ]:


df_result = (
    pd.concat([df_fuzzy_matches, df_exact_matches], ignore_index=True)
)

# display stats
print('Match ratio : {:.3f}'.format(df_result['n'].sum() / df_test_data['n'].sum()))
print('Categories: {:,} -> {:,}'.format(df_result['job_title_raw'].nunique(), df_result['job_title'].nunique()))

# 10 most transformed job titles 
(
    df_result
    .groupby('job_title')
    .size()
    .sort_values(ascending=False)
    .head(10)
)

