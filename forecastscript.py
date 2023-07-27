import pandas as pd
import scipy
import numpy as np
import sklearn
import sklearn.metrics
import datetime as dt
from pandas.tseries.offsets import MonthEnd
from collections import namedtuple
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
 
# %%
# INTERPOLATE AND SMOOTH A TIME SERIES (e.g., for a weight chart or CGM)
def make_interpolated_ts(df_input,
                         time_col,
                         data_col,
                         ds_res,
                         method='linear',
                         final_ds_res=None,
                         order=1,
                         as_integer=False,
                         extract_date_freq=None,
                         extract_date_label='target_date',
                         show_plot=False):
    """
    Rationale:
        - Interpolate a time series of values to a desired ds_res (e.g., daily values)
    Inputs:
        - df_input: dataframe with original data
        - time_col: string label of the column that captures time/date
        - data_col: the to-be-interpolated values
        - ds_res: period at which to resample
            - 'M' for monthly, 'D' for daily, 'T' for minute, '5T' for 5 minutes, etc.
        - method: e.g., 'linear', 'polynomial', 'time', 'spline', 'pad';
            see (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html)
        - final_ds_res: if not None, a final downsampling step (e.g.: ds_res='T' and final_ds_res='5T')
        - order: default is 1 for linear, otherwise an integer
        - as_integer: whether to force the interpolated series to be integer (rounded)
    Outputs:
        - resamp: the interpolated data, as a pandas series
        - an optional plot
    """
 
    # make a copy
    df_input = df_input.copy()
 
    # force timestamp
    df_input[time_col] = pd.to_datetime(df_input[time_col])
 
    # get first and last
    min_date = np.min(df_input[time_col])
    max_date = np.max(df_input[time_col])
 
    # set timestamp column as index
    df_input = df_input.set_index(time_col)
 
    # get the series we want
    data = df_input[data_col]
 
    # unsample; cf. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
    # note that the .mean() is critical here; unseen timestamps will have value of NaN
    unsamp = data.resample(ds_res).mean()
 
    # now we can resample through the NaN points
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
    resamp = unsamp.interpolate(method, order=order)
    
    # optional final downsampling step; use .first() here
    if final_ds_res is not None:
        resamp = resamp.resample(final_ds_res).first()
 
    if as_integer is True:
        # round it and force integer
        resamp = pd.to_numeric(np.round(resamp), downcast='integer')
 
    # optional: extract specific dates
    if extract_date_freq is not None:
 
        # target dates can only be WITHIN the date range of the original data
        tar_dates = pd.date_range(min_date, max_date, freq=extract_date_freq)
 
        yy = pd.DataFrame(tar_dates, columns=[extract_date_label])
        zz = pd.DataFrame(resamp).reset_index()
 
        # rename for clarity
        interp_col = '{}_interp'.format(data_col)
        zz.rename(columns={data_col: interp_col}, inplace=True)
 
        output = yy.merge(
            zz, how='left', left_on=extract_date_label, right_on=time_col)
 
        # drop the redundant column
        output.drop(columns=time_col, inplace=True)
 
    else:
        output = resamp
 
    if show_plot is True:
        figsize(16, 5)
 
        # the interpolated series
        plt.plot_date(resamp.index, resamp.values,
                      'r.', ms=6, label='Interpolated')
 
        # highlight the original data points on top
        plt.plot_date(data.index, data.values,
                      'k.', ms=6, label='Original')
 
        if extract_date_freq is not None:
            # highlight the extracted data points
            plt.plot_date(output[extract_date_label],
                          output[interp_col],
                          'm.', ms=8, label='Extracted')
 
        plt.grid()
        plt.legend()
        plt.tight_layout()
 
    # return a time series object
    return output
 

def find_period_bounds(date_str):
    """
    Purpose:
        - Get the start and end date for the "period" bounds associated with a given yyyy-mm-dd
    Notes:
        - date_str is a string date in yyyy-mm-dd format
        - Will throw an error if the date is not valid
        - cf. (https://www.pythonprogramming.in/how-to-get-start-and-end-of-week-data-from-a-given-date.html)
    Author:
        - RJ Ellis, August 2019
    """
 
    date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
 
    # start of week = a Monday
    sow = date_obj - dt.timedelta(days=date_obj.weekday())  # Monday
 
    # end of week = a Sunday
    eow = sow + dt.timedelta(days=6)  # Sunday
 
    sow = pd.to_datetime(sow)
    eow = pd.to_datetime(eow)
 
    # get year and month
    y = date_obj.year
    m = date_obj.month
 
    # start of month; can't do -MonthStart() here, do it manually
    som = pd.to_datetime(dt.datetime(y, m, 1))
 
    # end of month is easier
    eom = pd.to_datetime(date_str) + MonthEnd(0)
 
    # return as namedtuple for clarity
    Res = namedtuple('dates', 'sow eow som eom')
 
    return Res(sow, eow, som, eom)
 
def misc_normalize_counts_per_pd(df_input,
                                 period_col,
                                 ts_col,
                                 metrics=[]):
    """
    Purpose:
        - Normalize the raw counts of events to a given period; e.g., day, week, month
    Inputs:
        - df_input: dataframe with all data
        - period_col: string label of column specifying the period to
            normalize on (e.g., dates)
        - ts_col: string label of column with timestamps
        - metrics: a list of columns of data to normalize
    Output:
        - a new dataframe with normalized data (*_prc_pd)
    """
 
    print('Normalizing counts per week, to detect if there are DOW difffereces.')
 
    # get sum per period_col
    mo = df_input.groupby(period_col)[metrics].sum().reset_index()
 
    # rename for clarity
    new_names = [(i, i+'_sum_pd') for i in mo.iloc[:, 1:].columns.values]
    mo.rename(columns=dict(new_names), inplace=True)
 
    # merge
    temp = df_input.merge(mo, on=period_col)
 
    for m in metrics:
        temp['{}_prc_pd'.format(m)] = 100 * temp[m] / temp['{}_sum_pd'.format(m)]
 
    # order by ts for clarity
    temp.sort_values(by=ts_col, inplace=True)
    temp.reset_index(drop=True, inplace=True)
 
    # make sure it worked; test first metric
 
    # sum per week and round for simplicity
    xx = temp.groupby(period_col)['{}_prc_pd'.format(metrics[0])].sum().round()
 
    if (np.min(xx) == 100) & (np.max(xx) == 100):
        print(
            'After normalization, normalized hourly values correctly sum to 100% each week.')
    else:
        print('Warming: After normalization, normalized hourly values sum from {} to {} across weeks.'.format(np.min(xx), np.max(xx)))
 
    return temp
 
def find_us_holidays_per_date(df_input,
                              col_date,
                              col_dow=None,
                              replace_hol_with=8,
                              verbose=2):
    """
    Purpose:
        - Encode dates that are US holidays with a specific label for later operations
    Inputs:
        - df_input: a dataframe
        - col_date: string label of column that encodes date; 
            data can be string or datetime/Timestamp
        - col_dow: string label of column that encodes day of week;
            assumes values are <= 7; if None, will add column "dow", with 1 = Monday and 7 = Sunday
        - replace_hol_with: numeric value to indicate a holiday; default = 8
    Author:
        - RJ Ellis, August 2019
    """
 
    # import this module here, so it doesn't throw an error if not installed otherwise
    import holidays
 
    df_output = df_input.copy()
 
    # dates to test
    test = df_output[col_date]
    
    # do we need to add a DOW column?
    if col_dow is None:
        col_dow = 'dow'
 
        df_output[col_dow] = test.dt.weekday + 1  # add 1 so 1 = Monday and 7 = Sunday
 
    # get the holiday dates
    us_holidays = holidays.US()
 
    is_holiday = pd.Series([i in us_holidays for i in test])
 
    if verbose > 0:
        print('{} rows of data fall on a US holiday. Replacing values in {} with {}.'.format(np.sum(is_holiday),
                                                                                             col_dow, replace_hol_with))
 
    # add column
    df_output['is_holiday'] = is_holiday
 
    if replace_hol_with is not None:
 
        # replace day of week
        df_output[col_dow].where(
           ~df_output['is_holiday'], replace_hol_with, inplace=True)
 
    return df_output
 

def misc_est_24_hour_by_dow(df_input,
                            data_metric,
                            col_date,
                            col_dow,
                            col_hour,
                            add_holidays=False,
                            estimator='percentile',
                            percentile=50,
                            min_points=2,
                            lw=2,
                            show_plot=True,
                            verbose=2):
    """
    Rationale:
         - This custom function allows for plotting specific percentiles; sns.lineplot does not permit this
         - cf. https://github.com/mwaskom/seaborn/issues/1501
    Inputs:
        - df_input: a dataframe
        - data_metric: string label of variable to be plotted
        - col_dow: string label of column specifying day of week (1 = Monday, 7 = Sunday)
        - col_hour: string label of column specifying hour of day (0 through 23)
        - estimator: either 'mean' or 'percentile'
        - percentile: numeric value from 0 to 100
        - min_points: minimum number of data points per hour in order to show a point estimate
        - lw: line width
        - show_plot: if False, don't make a plot (helpful if using this function as a helper)
    Outputs:
        - a plot
        - a dataframe: point estimates for each DOW and hour of day
    """
 
    # calculate the desired metrics
    key_percentiles = [percentile/100]
 
    # optional: convert holidays to day of week = 8
    if add_holidays is True:
        df_input = find_us_holidays_per_date(df_input,
                                             col_date=col_date,
                                             col_dow=col_dow,
                                             replace_hol_with=8,
                                             verbose=verbose)
 
    # get the percentile(s)
    data_prc = df_input.groupby([col_dow, col_hour])[data_metric].describe(
        percentiles=key_percentiles).reset_index()
 
    data_prc.drop(columns=['std', 'max'], inplace=True)
 
    # call the column carefully!
    if estimator == 'percentile':
        if percentile == 50:
            est = '50%'
        elif percentile < 50:
            est = data_prc.columns[-2]  # for safety, call it programmatically
        elif percentile > 50:
            est = data_prc.columns[-1]  # for safety, call it programmatically
        ylab = ylab = '{}th percentile of volume'.format(percentile)
 
    elif estimator == 'mean':
        est = 'mean'
        ylab = 'Mean volume'
    elif estimator == 'count':
        est = 'count'
        ylab = 'Number of dates'
 
    # add a specific column for clarity
    data_prc['target'] = data_prc[est]
 
    if show_plot is True:
 
        # get the min and max counts
        min_count = int(np.min(data_prc['count']))
        max_count = int(np.max(data_prc['count']))
 
        print('Number of dates per day of week and hour: {} to {}'.format(
            min_count, max_count))
 
        # set things up for plot
        max_dow = np.max(data_prc[col_dow])
 
        # start with basic
        dow = [1, 2, 3, 4, 5, 6, 7]
        dow_labels = 'Mon Tue Wed Thu Fri Sat Sun'.split()
        colors = ['red', 'darkorange', 'gold',
                  'green', 'royalblue', 'violet', 'purple']
 
        if max_dow == 8:
            dow.append(8)
            dow_labels.append('Hol')
            colors.append('black')
 
        figsize(16, 5.5)
 
        for d in np.arange(len(dow)):
            look1 = data_prc[col_dow] == dow[d]
            look2 = data_prc['count'] >= min_points
 
            temp2 = data_prc[look1 & look2]
            plt.plot(temp2[col_hour],
                     temp2['target'],
                     lw=lw,
                     color=colors[d],
                     label=dow_labels[d])
 
        plt.legend(loc=(1.01, 0), title='DOW')
        xticks = np.arange(0, 24)
        plt.xticks(xticks, xticks)
        plt.xlim(0, 23)
        plt.ylim(bottom=0)
        plt.xlabel('Hour of day (Central time)')
 
        plt.ylabel(ylab)
 
        plt.grid(axis='y')
        plt.tight_layout()
 
    return data_prc  # will use this as input to other functions
 
def calc_mase(y_train, y_test, y_pred, decimals=5):
    
    """
    Purpose:
        - Compute MASE: Mean Absolute Scaled Error, proposed by Rob Hyndman
        - Interpretation: Values of MASE greater than one indicate that the forecast errors (i.e., abs(y_pred - y_test)) are worse, 
            on average, than in-sample one-step forecasts from the naive method (i.e., first-order difference of y_train).
    Inputs:
        - y_train: array-like; training data (seen by model)
        - y_test: array-like; test data (not seen by model)
        - y_pred: array-like; model predictions
    References:
        - Overview: https://robjhyndman.com/papers/foresight.pdf
        - Preprint: https://robjhyndman.com/papers/mase.pdf
        - Citation: https://www.sciencedirect.com/science/article/abs/pii/S0169207006000239
    Other implementations:
        - https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
        - https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/TimeSeries/MASE.py
    Author:
        - RJ Ellis, Apr 2022
    """
    
    # convert to arrays to be safe (i.e., in case index locations differ)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
        
    n = len(y_train)
 
    # numerator: mean absolute error between y_pred and y_test; will throw a ValueError if these arrays do not have the same length, which is helpful
    numer = np.nanmean(np.abs(y_pred - y_test))
 
    # denominator: mean absolute first-order difference in training data
    denom = (1/(n-1))*np.nansum(np.abs(np.diff(y_train)))
 
    return np.round(numer / denom, decimals)
 
def calc_regression_metrics(
    y_true,
    y_pred, 
    y_train=None,
    pct_thr=5,
    total_sum=False, 
    decimals=4,
    verbose=True
):
    '''
    Purpose:
        - Compute standard accuracy metrics based on two arrays: y_true and y_pred
    Inputs:
        - y_true: array-like (i.e., y_test data, not seen by model)
        - y_pred: array-like (model predictions)
        - y_train: optional; used to compute MASE; if not available (i.e., y_train=None, MASE will return NaN)
            * MASE is only relevant for temporal forecast problems, where y_train is a meaninfully ordered series
            of y-axis values as a function of time (implied on the x-axis)
        - pct_thr: to compute "the proportion of (y_true, y_pred) pairs that have an abs pct error <= pct_thr" metric
        - total_sum: if True, compare the total sum of all values in y_true and y_pred
            * only set True when the underlying data is *counts*, and those counts are not cumulative
            * this is potentially useful for, e.g., a model that is predicting number of counts on y for a given input x
        - verbose: if True, print the stats
    Outputs:
        - Metrics will display on screen if selected
        - Also returns a namedtuple of results
    Notes:
        - Refer to R2 as "CD" (coefficient of determination) for clarity, since the value can be negative
        - See a detailed review in Chicco et al 2021, who find that R2 is the "most informative" metric to report
            * https://pubmed.ncbi.nlm.nih.gov/34307865/ 
            * https://www.readcube.com/articles/10.7717/peerj-cs.623 
    References:
        - https://en.wikipedia.org/wiki/Coefficient_of_determination
        - https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    Author:
        - RJ Ellis, 2021
    '''
 
    # convert to arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
 
    # remove nans
    keep1 = ~np.isnan(y_true)
    keep2 = ~np.isnan(y_pred)
    keep = keep1 & keep2
 
    y_true = y_true[keep]
    y_pred = y_pred[keep]
 
    num_pts = len(y_true)
 
    # ------------
    # compute metrics; label "r2_score" as "cd" for coef of determination, for clarity
    ev = sklearn.metrics.explained_variance_score(y_true, y_pred).round(decimals)
    cd = sklearn.metrics.r2_score(y_true, y_pred).round(decimals)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred)).round(decimals)
    mean_ae = sklearn.metrics.mean_absolute_error(y_true, y_pred).round(decimals)
    med_ae = sklearn.metrics.median_absolute_error(y_true, y_pred).round(decimals)
 
    # mean absolute percentage error
    # https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn
 
    # ignore (but report) cases where y_true == 0
    ex = y_true == 0
    num_zero = sum(ex)
 
    if verbose and num_zero > 0:
        print('Note: There are {:,} instances of y_true == 0.'.format(num_zero))
        print('-----')
 
    mean_ape = np.mean(100*(np.abs(y_true[~ex] - y_pred[~ex]) / y_true[~ex])).round(decimals)
 
    # median absolute percentage error
    # will help mitigate if there are very low true counts that would have high percentage error
    med_ape = np.median(100*(np.abs(y_true[~ex] - y_pred[~ex]) / y_true[~ex])).round(decimals)
 
    # newly added wape
    weighted_ape = (np.sum(np.abs(y_true[~ex] - y_pred[~ex])) / np.sum(y_true[~ex])).round(decimals)
 
    # MASE: mean absolute scaled error (proposed by Rob Hyndman)
    if y_train is not None:
        mase = calc_mase(y_train=y_train, y_test=y_true, y_pred=y_pred, decimals=decimals)
    else:
        mase = np.nan
 
    # optional: compute the total number of actual and predicted in test data only
    if total_sum is True:
        y_true_sum = int(np.ceil(np.sum(y_true)))
        y_pred_sum = int(np.ceil(np.sum(y_pred)))
        prc_err_sum = np.round(100*(y_pred_sum - y_true_sum)/y_true_sum, 2)
 
        # show sign for clarity
        if prc_err_sum > 0:
            sign = '+'
        else:
            sign = ''  # negative sign is already included
 
    else:
        y_true_sum = np.nan
        y_pred_sum = np.nan
        prc_err_sum = np.nan
 
    if verbose > 0:
        print('Regression performance metrics ({} data points):'.format(num_pts))
        print('   Coefficient of determination (CD or R2):', np.round(cd, decimals))
        print('   Explained variance (EV):', ev)
        print('   RMSE:', rmse)
        print('   Mean AE:', mean_ae)
        print('   Mean APE (as %):', mean_ape)
        print('   Median AE:', med_ae)
        print('   Median APE (as %):', med_ape)
        print('   WAPE:', weighted_ape)
 
        # MASE
        if y_train is not None:
            print('   MASE:', mase)
        else:
            print('   MASE: (not computed)')
 
        if total_sum is True:
            print('Total event counts:')
            print('   Actual: {:,}'.format(y_true_sum))
            print('   Predicted: {:,}'.format(y_pred_sum))
            print('   Error: {}{}%'.format(sign, prc_err_sum))
 
    # -------------------------------
    # store metrics as a named tuple
 
    # standard metrics
    Metrics = namedtuple('Metrics', 'num_pts cd ev rmse mean_ae med_ae mean_ape med_ape weighted_ape mase')
    metrics = Metrics(num_pts, cd, ev, rmse, mean_ae, med_ae, mean_ape, med_ape, weighted_ape, mase)
 
    # optional other metrics
    if total_sum is True:
        Metrics2 = namedtuple('Metrics2', 'y_true_sum y_pred_sum prc_err_sum')
        metrics2 = Metrics2(y_true_sum, y_pred_sum, prc_err_sum)
 
        # now, *combine* with original tuple
        # https://stackoverflow.com/questions/12161649/what-is-the-simple-way-to-merge-named-tuples-in-python
        Metrics = namedtuple('Metrics', Metrics._fields + Metrics2._fields)
        metrics = Metrics(*metrics, *metrics2)
 
    return metrics
 
def model_nested_forecast(df_period_fcst,
                          df_hourly,
                          hourly_pd_col,
                          hourly_dt_col,
                          hourly_dow_col,
                          hourly_hr_col,
                          hourly_counts_col,                        
                          add_holidays=False,
                          num_test_pds=0,
                          make_future=False,
                          estimator='percentile',
                          percentile=50,
                          true_count_thr=0,
                          verbose=2):
    """
    Rationale:
        - Perform "Nested Forecasting Model" (NFORM):
            1. Take an *already made* Prophet forecast that projects a desired metric 
               into the future. This dataframe must contain these columns:
                - 'ds' (timestamp), 
                - 'y' (actual values)
                - 'yhat' (predicted values, including future values).
            2. Take an "hourly level" dataframe that has hour by hour event counts.
            3. Normalize those hourly counts each week. Optionally code in days that are holidays.
            4. Take a single point estimate (mean or specified percentile) 
                across all weeks. This is the "hourly level estimate".
            5. Convolve (i.e., multiply) the hourly level estimate with the 
                Prophet-forecasted weekly tallies.
    Inputs:
        - df_period_fcst: dataframe with weekly-level event counts
        - df_hourly: dataframe with hourly-level event counts
            * assumes the following columns are present: _dt, _eow
        - Columns within df_hourly:
            - hourly_pd_col: string label that contains distinct days/weeks/months; 
                must match dates within df_period_fcst
            - hourly_dt_col: string label of distinct dates
            - hourly_dow_col: string label of day of week values (1= Monday, 7= Sunday)
            - hourly_hr_col: string label of hour values (integers, 0 through 23)
            - hourly_counts_col: string label of counts in df_hourly; the to-be-modeled event counts
        - add_holidays: False (default) or True
        - num_test_pds: 0 by default; if > 0, without the last N periods as test data
        - make_future: if True, and if there are future dates in df_period_fcst 
            (relative to df_hourly), make hourly forecast into the future.
        - estimator: either 'mean' or 'percentile'
        - percentile: a *single* value from 0 to 100
        - true_count_thr: exclude data points where true count is less than this value
            prior to computing accuracy metrics
        - verbose: if 0, no text output on screen
    Outputs:
        - Three dataframe objects; in order:
            [0] Fully merged dataframe with all data
            [1] Test data only (if performing train-test split, otherwise = None)
            [2] Key statistics
        - (Plotting results is handled in a separate function)
    Author:
        - RJ Ellis, August 2019
    """
 
    df_hourly = df_hourly.copy()
 
    if verbose > 0:
        print('Input dataframe has {} rows of data.'.format(len(df_hourly)))
 
    if num_test_pds == 0:
        # all the data is "training"
        train = df_hourly
 
    elif num_test_pds > 0:
 
        # determine the split point week
        udates = df_hourly[hourly_pd_col].unique()
        split_date = str(np.min(udates[-num_test_pds-1:]))[:10]
 
        if verbose > 0:
            print('Spliting data into train and test at {}.'.format(split_date))
 
        ind_train = df_hourly[hourly_pd_col].between('1900-01-01', split_date)  # date in far past
        ind_test = df_hourly[hourly_pd_col].between(split_date, '2100-01-01')  # date in far future
 
        # add in index locations of test data
        df_hourly['is_test'] = ind_test
 
        train = df_hourly[ind_train]
        train_dates = train[hourly_pd_col].nunique()
 
        # print confirmation
        if verbose > 0:
            print('Of the {} weeks of data, reserving {} for training and {} for testing.'.format(len(udates),
                                                                                                  train_dates,
                                                                                                  num_test_pds))
    # -----------
    # compute the desired percentile on the in-sample data
    norm_counts_col = '{}_prc_pd'.format(hourly_counts_col)
 
    percentile_res = misc_est_24_hour_by_dow(df_input=train, 
                                             data_metric=norm_counts_col,
                                             col_date=hourly_dt_col,
                                             col_dow=hourly_dow_col,
                                             col_hour=hourly_hr_col,
                                             add_holidays=add_holidays,
                                             estimator=estimator,
                                             percentile=percentile,
                                             show_plot=False,
                                             verbose=verbose)
 
    # alias the column that contains the target percentile
    prc_column = 'target'
 
    if make_future is True:
        # here, we assume that the weekly dataframe extends into
        # the future relative to the hourly data
 
        # get last date in hourly (assume yyyy-mm-dd)
        f_start = np.max(df_hourly[hourly_dt_col])
 
        # get last date in weekly (assume yyyy-mm-dd)
        f_end = np.max(df_period_fcst['ds'])
 
        # create a future hourly series
        future_hourly = pd.date_range(f_start, f_end, freq='H')
 
        # extract dates as strings
        future_date_str = [str(i)[:10] for i in future_hourly]
 
        # get *end of week* for each date
        future_eow = [find_period_bounds(i)[1] for i in future_date_str]
 
        # -----
        # new dataframe
        df_future = pd.DataFrame()
        df_future['ts'] = future_hourly
        df_future['hour'] = df_future['ts'].dt.hour
        df_future['date'] = pd.to_datetime(future_date_str)
        df_future['dow'] = df_future['date'].dt.weekday + 1
        df_future['period_end'] = pd.to_datetime(future_eow)
 
        # -----
        # merge in quantile results
        merged = df_future.merge(percentile_res, how='left',
                                 left_on=['dow', 'hour'],
                                 right_on=[hourly_dow_col, hourly_hr_col])
 
        # -----
        # merge in future forecast weekly counts
        temp1 = df_period_fcst[['ds', 'yhat']].copy()
        temp1.rename(columns={'yhat': 'pd_events_pred'}, inplace=True)
 
        merged = merged.merge(
            temp1, how='left', left_on='period_end', right_on='ds')
 
        # clean up
        merged.drop(columns='ds', inplace=True)
 
        # for output formatting correctly
        stats = []
        merged_test = []
 
    elif make_future is False:
        # here, we are either working with train/test data,
        # or just data from train; no future estimation
 
        # -----
        # merge quantile results onto df_hourly
        merged = df_hourly.merge(percentile_res, how='left', on=[
                                 hourly_dow_col, hourly_hr_col])
 
        # -----
        # merge in weekly forecast; we want yhat, which is predicted number of events per week
        temp1 = df_period_fcst[['ds', 'y', 'yhat']].copy()
        temp1.rename(columns={'y': 'pd_events_true',
                              'yhat': 'pd_events_pred'}, inplace=True)
 
        merged = merged.merge(
            temp1, how='left', left_on=hourly_pd_col, right_on='ds')
 
    # -----
    # compute the *predicted* raw counts each hour and day of week
    # we multiply each percentage value by the *predicted* weekly events total from the weekly forecast
    pred_label = '{}_pred'.format(hourly_counts_col)
 
    # round up
    # pred_vals = np.ceil(merged[prc_column] / 100 * merged['pd_events_pred'])
    pred_vals = merged[prc_column] / 100 * merged['pd_events_pred']
 
    # store it
    merged[pred_label] = pd.to_numeric(pred_vals, downcast='integer')
 
    if make_future is False:
 
        # --------------
        # EVALUATE MODEL ACCURACY
 
        if num_test_pds == 0:
            ytrue = merged[hourly_counts_col]
            ypred = merged[pred_label]
 
            merged_test = None  # so output is formatted correctly
 
            if verbose > 0:
                print('-------')
                print('Accuracy metrics below are for the ENTIRE dataset.')
 
        elif num_test_pds > 0:
 
            # isolate the test data
            merged_test = merged[ind_test]
 
            ytrue = merged_test[hourly_counts_col]
            ypred = merged_test[pred_label]
 
            # optional: limit data based on true_count_thr
            if true_count_thr > 0:
                look = ytrue >= true_count_thr
                ytrue = ytrue[look]
                ypred = ypred[look]
 
            if verbose > 0:
                print('-------')
                print('Accuracy metrics below are for the TEST dataset.')
                display_stats = True
            else:
                display_stats = False
 
        # we either compute stats on all data (if no test) or test data (if specified)
        stats = calc_regression_metrics(
            ytrue, ypred,
            verbose=display_stats,
            total_sum=True)
 
    return merged, merged_test, stats
 

def model_nested_hyperparm_tuning(df_period_fcst,
                                  df_hourly,
                                  hourly_pd_col,
                                  hourly_dt_col,
                                  hourly_dow_col,
                                  hourly_hr_col,
                                  hourly_counts_col,
                                  add_holidays,
                                  num_test_pds,
                                  true_count_thr,
                                  percentiles):
 
    # the stats
    cd = []
    mean_ae = []
    med_ae = []
    mean_ape = []
    med_ape = []
    w_ape = []
    rmse = []
    prc_err_sum = []
 
    print('Working ...')
 
    for p in percentiles:
 
        # run each model
        zz, zz, stats = model_nested_forecast(df_period_fcst=df_period_fcst,
                                              df_hourly=df_hourly,
                                              hourly_pd_col=hourly_pd_col,
                                              hourly_dt_col=hourly_dt_col,
                                              hourly_dow_col=hourly_dow_col,
                                              hourly_hr_col=hourly_hr_col,
                                              hourly_counts_col=hourly_counts_col,
                                              add_holidays=add_holidays,
                                              num_test_pds=num_test_pds,
                                              make_future=False,
                                              estimator='percentile',
                                              percentile=p,
                                              true_count_thr=true_count_thr,
                                              verbose=0)
 
        # store the stats
        cd.append(stats.cd)
        mean_ae.append(stats.mean_ae)
        med_ae.append(stats.med_ae)
        mean_ape.append(stats.mean_ape)
        med_ape.append(stats.med_ape)
        w_ape.append(stats.weighted_ape)
        rmse.append(stats.rmse)
        prc_err_sum.append(stats.prc_err_sum)
 
    # dataframe
    res = pd.DataFrame()
    res['percentile'] = percentiles
    res['CD'] = cd
    res['MeanAE'] = mean_ae
    res['MeanAPE'] = mean_ape
    res['WAPE'] = w_ape
    res['RMSE'] = rmse
    res['test_prc_err'] = prc_err_sum
 
    # geomean of MeanAE, MeanAPE, RMSE
    cols = 'MeanAE MeanAPE RMSE'.split()
    temp = res[cols]
    res['geomean_err'] = scipy.stats.hmean(temp, axis=1)
 
    return res