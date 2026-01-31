import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.metrics import concordance_index_censored as concordance

from tabicl.sklearn.surver import TabICLSurver


def stratified_group_split(df: pd.DataFrame,
                           group_col: str ='pid',
                           stratify_col: str ='event',
                           test_frac: float = 0.2,
                           seed: int | None = None):
    """
    Function to make sure that patient IDs don't show up in both training and test sets. And then also balance the event rate across the two sets.
    """
    # Step 1: Collapse to group-level (one row per patient)
    group_df = df.groupby(group_col)[stratify_col].any().astype(int).reset_index()
    # Step 2: Stratified split on the group level
    group_train, group_test = train_test_split(
        group_df,
        stratify=group_df[stratify_col],
        test_size=test_frac,
        random_state=seed
    )
    # Step 3: Merge back to original data
    df_train = df[df[group_col].isin(group_train[group_col])]
    df_test = df[df[group_col].isin(group_test[group_col])]
    return df_train, df_test


def bootstrap_concordance_index(
    df: pd.DataFrame,
    id_col: str = "id",
    event_col: str = "event",
    time_col: str = "time",
    score_col: str = "score",
    n_bs: int = 1000,
    alpha: float = 0.05,
    ) -> pd.DataFrame:
    """
    Wrapper on the concordance_index_timevarying function to perform bootstrap resampling for confidence intervals.
    """
    # Input checks
    expected_cols = [id_col, event_col, time_col, score_col]
    missing_cols = np.setdiff1d(expected_cols, df.columns)
    assert len(missing_cols) == 0, f"Missing required columns: {missing_cols.tolist()}"
    assert isinstance(n_bs, int) and n_bs > 0, "n_bs must be a positive integer."
    assert 0 < alpha < 1, "alpha must be between 0 and 1."
    # Subset to only the required columns
    df = df[expected_cols].copy()
    # Set up the storage holder and final DataFrame slice
    holder_bs = np.zeros(n_bs)
    # Get baseline result and then loop over the bootstrap samples
    conc_test = concordance(df[event_col].astype(bool), df[time_col], df[score_col])[0]
    # Bootstrap
    for j in range(n_bs):
        res_bs = df.groupby(event_col).sample(frac=1,replace=True,random_state=j)
        conc_bs = concordance(res_bs[event_col].astype(bool), res_bs[time_col], res_bs[score_col])[0]
        holder_bs[j] = conc_bs
    # Add on the baseline result and empirical confidence intervals
    lb, ub = np.quantile(holder_bs, [alpha,1-alpha])
    holder_cindex = pd.DataFrame(np.atleast_2d((conc_test, lb, ub)), columns=['cindex', 'lb', 'ub'])
    return holder_cindex


def main():
    # Standard imports
    import os
    # sksurv imports
    from sksurv.util import Surv as surv_util
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble.forest import RandomSurvivalForest
    # sklearn imports for data preprocessing
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import make_column_selector, ColumnTransformer
    # SurvSet package imports
    from SurvSet.data import SurvLoader

    ###############################
    # --- (1) PARAMETER SETUP --- #

    # Save to the examples directory
    dir_base = os.getcwd()
    dir_sim = os.path.join(dir_base, 'simulation', )
    print('Figure will saved here: %s' % dir_sim)

    # Concordance empirical alpha level
    alpha = 0.1
    # Number of bootstrap samples
    n_bs = 250
    # Set the random seed
    seed = 1234
    # Percentage of data to use for testing
    test_frac = 0.3

    #####################################
    # --- (2) ENCODER/MODEL/LOADER --- #

    # (i) Set up feature transformer pipeline
    # enc_fac = Pipeline(steps=[('ohe', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'))])
    enc_fac = Pipeline(steps=[('ohe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1))])
    sel_fac = make_column_selector(pattern='^fac\\_')
    enc_num = Pipeline(steps=[('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())])
    sel_num = make_column_selector(pattern='^num\\_')
    # Combine both
    enc_df = ColumnTransformer(transformers=[('ord', enc_fac, sel_fac), ('s', enc_num, sel_num)])
    enc_df.set_output(transform='pandas')  # Ensure output is a DataFrame

    # (ii) Run on datasets
    senc = surv_util()
    loader = SurvLoader()

    ##################################
    # --- (3) LOOP OVER DATASETS --- #

    # (i) Initialize results holder and loop over datasets
    n_ds = len(loader.df_ds)
    holder_cindex = []
    for i, r in loader.df_ds.iterrows():
        is_td, ds = r['is_td'], r['ds']
        if is_td:
            continue  # We're only interested in static datasets
        print('Dataset %s (%i of %i)' % (ds, i + 1, n_ds))
        df = loader.load_dataset(ds)['df']
        # Split based on both the event rate and unique IDs
        df_train, df_test = stratified_group_split(df=df, group_col='pid',
                                                   stratify_col='event', test_frac=test_frac, seed=seed)
        assert not df_train['pid'].isin(df_test['pid']).any(), \
            'Training and test sets must not overlap in patient IDs.'
        # Fit encoder
        enc_df.fit(df_train)
        # Transform data
        # X_train = df_train.drop(columns=['pid', 'event', 'time'])
        X_train = enc_df.transform(df_train)
        # assert X_train.columns.str.split('\\_{1,2}', expand=True).to_frame(False)[1].isin(
        #     ['fac', 'num']).all(), 'Expected feature names to be prefixed with "fac_" or "num_"'
        # X_test = df_test.drop(columns=['pid', 'event', 'time'])
        X_test = enc_df.transform(df_test)
        if df.shape[0] > 1024 or X_train.shape[1] > 200 or X_test.shape[1] > 200:
            continue  # For now, TabICL only supports up to 1024 rows and 200 features
        # Set up the survival object and fit the model
        mdl = TabICLSurver()
        # mdl = RandomSurvivalForest()
        # mdl = CoxnetSurvivalAnalysis(normalize=True)
        # Set up Surv object for static model and fit
        So_train = senc.from_arrays(df_train['event'].astype(bool), df_train['time'])
        mdl.fit(X=X_train, y=So_train)
        # Get test prediction
        scores_test = mdl.predict(X_test)
        # Prepare test data for concordance calculation
        res_test = df_test[['pid', 'event', 'time']].assign(scores=scores_test)
        if is_td:
            res_test['time2'] = df_test['time2'].values
        # Generate results and bootstrap concordance index
        res_cindex = bootstrap_concordance_index(res_test, 'pid', 'event', 'time', 'scores', n_bs, alpha)
        res_cindex.insert(0, 'ds', ds)
        res_cindex.insert(1, 'rows', r['n'])
        res_cindex.insert(2, 'cat', r['n_fac'])
        res_cindex.insert(3, 'num', r['n_num'])
        res_cindex.insert(3, 'censor', df['event'].sum() / df['event'].shape[0])
        holder_cindex.append(res_cindex)

    # (ii) Merge results
    df_cindex = pd.concat(holder_cindex, ignore_index=True, axis=0)
    ds_ord = df_cindex.sort_values('cindex')['ds'].values
    df_cindex['ds'] = pd.Categorical(df_cindex['ds'], ds_ord)
    df_cindex.to_csv('cindex.csv', index=False)

    print('~~~ The SurvSet.sim_run module was successfully executed ~~~')


if __name__ == '__main__':
    # Call the main module
    main()