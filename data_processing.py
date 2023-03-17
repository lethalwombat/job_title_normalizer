import pandas as pd
from rapidfuzz import process
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=4)


def standardise_job_title(df: pd.DataFrame, dictionary=None, col='job_title', score_cutoff=95) -> pd.DataFrame:
    """
    Standardise job titles for feature engineering. Takes in a DataFrame and return a DataFrame with the standard column.

        Parameters:
            df  (pd.DataFrame): Input data
            col (str): Input column (the default is 'job_title')
            score_cutoff (int): Fuzzy matching score cutoff (the default is 95)            
        Returns:
            df (pd.DataFrame): DataFrame with the standardised job title
    """
    # nothing to look up, no column found or empty dataframe as input
    if any(
        [dictionary is None, col not in df.columns, df.shape[0] == 0]
    ):
        print('No job title dictionary set, no job title column found or empty data input!')
        return df
    # read in job titles dictionary
    dictionary = (
        dictionary
        .rename(columns={'job_title_raw' : '_job_title_raw', 'job_title' : '_job_title'})
    )
    # create match keys
    df = (
        df
        .assign(
            _match_key=df[col].apply(lambda x : x.lower().strip() if isinstance(x, str) else '')
        )
    )
    # get exact matches
    matches_exact = (
        df
        .merge(dictionary, how='left', left_on='_match_key', right_on='_job_title_raw', indicator=True, suffixes=('', '_y'))
    )
    # where we could not find an exact matches, we will try to fuzzy match
    matches_fuzzy = (
        matches_exact
        .query('_merge == "left_only"')
        .drop(columns=['_merge', '_job_title_raw', '_job_title'])
    )
    # job titles to lookup as one array
    _job_titles = dictionary['_job_title_raw'].unique()
    
    # matching function wrapper
    _match_func = lambda x : process.extractOne(x, _job_titles)
    
    # apply matching function
    matches_fuzzy = (
        matches_fuzzy
        .assign(
            _match=matches_fuzzy['_match_key'].parallel_apply(_match_func)
        )
    )
    # split out fuzzy matching results into separate columns
    matches_fuzzy = (
        matches_fuzzy
        .assign(
            _match_value=matches_fuzzy['_match'].apply(lambda x : x[0]),
            _match_score=matches_fuzzy['_match'].apply(lambda x : x[1]),
        )
        .merge(dictionary, how='left', left_on='_match_value', right_on='_job_title_raw', suffixes=('', '_y'))
        .drop(columns=['_match'])
    )
    # split out non matches
    non_matches_fuzzy = (
        matches_fuzzy
        .query('_match_score < @score_cutoff')
    )
    # replace original value with the fuzzy match results
    matches_fuzzy[col] = matches_fuzzy['_job_title']

    # combine results
    matches = pd.concat([
        (non_matches_fuzzy
         [df.columns]
         .drop(columns=['_match_key'])), # no match found
        (matches_fuzzy
         .query('_match_score >= @score_cutoff')
         [df.columns]
         .drop(columns=['_match_key'])), # fuzzy match found,
        (matches_exact
         .assign(_temp=matches_exact['_job_title'])
         .drop(columns=[col])
         .rename(columns={'_temp' : col})
         .query('_merge == "both"')
         [df.columns]
         .drop(columns=['_match_key'])), # exact match found
    ], ignore_index=True)    
    # return combined results
    return matches
