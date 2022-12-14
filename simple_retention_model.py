import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from lifelines import WeibullFitter, ExponentialFitter
from weibull_mixture_fitter import WeibullInfiniteMixtureFitter
from docs import explanation

plt.style.use('dark_background')


def harmonic_survival(timelines):
    return 1/(timelines + 1)


def weibull_survival(timelines, lambda_, rho_):
    return np.exp(-np.power(timelines/lambda_, rho_))


def mixed_weibull_and_retained(timelines, lambda_, rho_, gamma):
    return gamma + (1-gamma)*weibull_survival(timelines, lambda_, rho_)


def expected_customers_weibull_mixture(n_periods, N_0, lambda_, rho_, gamma):
    retained = np.zeros(n_periods)
    periods = np.array([i for i in range(n_periods)])
    retention = mixed_weibull_and_retained(periods, lambda_, rho_, gamma)
    for n in range(1, n_periods):
        retained[n] = N_0 * sum(retention[:n])
    return retained


def expected_customer_growth_weibull_mixture(n_periods, N_0, lambda_, rho_, gamma):
    periods = np.array([i for i in range(n_periods)])
    retention = mixed_weibull_and_retained(periods, lambda_, rho_, gamma)
    return N_0 * retention


def expected_customers_harmonic(n_periods, N_0):
    retained = np.zeros(n_periods)
    periods = np.array([i for i in range(n_periods)])
    retention = harmonic_survival(periods)
    for n in range(1, n_periods):
        retained[n] = N_0 * sum(retention[:n])
    return retained


def expected_customer_growth_harmonic(n_periods, N_0):
    periods = np.array([i for i in range(n_periods)])
    retention = harmonic_survival(periods)
    return N_0 * retention


def convert_retention_cohorts_to_survival_data(retention_df):
    durations, observed = [], []
    for date, row in retention_df.iterrows():
        n_cohort = row[0]
        n_cohort_churned = 0
        for i, n_churned in enumerate(row.diff().abs()):
            if i > 0:  # skip first col since it's always a NaN
                if not np.isnan(n_churned):
                    for j in range(int(n_churned)):
                        durations.append(i)
                        observed.append(True)
                    n_cohort_churned += n_churned
                else:
                    # first NaN found after the first means no data for that cohort and duration
                    break
        for j in range(int(n_cohort - n_cohort_churned)):
            durations.append(i)
            observed.append(False)
    return durations, observed


def calculate_aggregate_retention(retention_df):
    """

    :param retention_cohort:
    :return:
    """
    retention = 100*retention_df.apply(lambda row: row / row[0], axis=1)
    weights = retention_df.apply(
        lambda col: retention_df[0][~col.isna()] / retention_df[0][~col.isna()].sum(),
        axis=0
    )
    return (retention * weights).sum(axis=0)


with st.sidebar:
    st.header('Model parameters')
    st.markdown('''
     You can upload your retention data as a CSV (in beta) or play with
     the parameters below to generate different scenarios:
     ''')

    uploaded_file = st.file_uploader(
        "Upload your retention data",
        help='The file must be CSV formatted with cohorts in its rows and periods from start as its columns')

    survival_model = st.selectbox(
        'Survival model',
        ('Exponential', 'Weibull', 'Mixture of Weibull and infinite (*)', 'Harmonic'),
        index=1)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=0, index_col=0)
        df.index.rename('cohort', inplace=True)
        df.rename(columns=lambda x: int(x), inplace=True)
        default_new_customers = round(df[0].mean())

        durations, observed = convert_retention_cohorts_to_survival_data(df)
        aggregate = calculate_aggregate_retention(df)

        if survival_model == 'Weibull':
            fitted_model = WeibullFitter().fit(durations, observed)
            lambda_default = float(round(fitted_model.lambda_, 2))
            rho_default = float(round(fitted_model.rho_, 2))
            pct_highly_retained_default = 0.
        elif survival_model == 'Mixture of Weibull and infinite (*)':
            fitted_model = WeibullInfiniteMixtureFitter().fit(durations, observed)
            lambda_default = float(round(fitted_model.lambda_, 2))
            rho_default = float(round(fitted_model.rho_, 2))
            pct_highly_retained_default = 100*float(round(fitted_model.p_, 2))
        elif survival_model == 'Harmonic':
            fitted_model = None
            lambda_default = 0
            rho_default = 1.0
            pct_highly_retained_default = 0.
        else:
            fitted_model = ExponentialFitter().fit(durations, observed)
            lambda_default = float(round(fitted_model.lambda_, 2))
            rho_default = 1.0
            pct_highly_retained_default = 0.

        n_periods_default = df[0].shape[0]*2
    else:
        df = None
        aggregate = None
        default_new_customers = 100
        lambda_default = 10.
        rho_default = 1.
        pct_highly_retained_default = 10.
        n_periods_default = 12

    n_periods = st.number_input('Number of periods', value=n_periods_default, min_value=6, max_value=120)
    avg_new_customers = st.number_input("New customers / period", min_value=10, value=default_new_customers, step=100)
    lambda_ = st.slider('Average time to churn', 0., 20., lambda_default, help='Larger values correspond to larger life times', step=1.)
    rho_ = st.select_slider('Churn rate deceleration/aceleration', options=sorted([0.01, 0.1, 0.5, 0.75, 1., 2., 5., 10., rho_default]), value=rho_default)
    pct_highly_retained = st.slider('% forever retained', 0., 100., pct_highly_retained_default, step=5.)

    agree = st.checkbox('Display % forever retained')

timeline = np.array([i for i in range(n_periods)])

if survival_model == 'Harmonic':
    retention_profile = harmonic_survival(timeline)
else:
    retention_profile = mixed_weibull_and_retained(timeline, lambda_, rho_, pct_highly_retained/100)

st.header("Simplified retention model")

tab1, tab2, tab3 = st.tabs(['Retention profile', 'Input data', 'Explanation'])

fig, ax = plt.subplots(figsize=(7, 3))

ax.plot(timeline, 100*retention_profile, '-ro')

if aggregate is not None:
    ax.bar(aggregate.index, aggregate.values)

if agree:
    ax.plot(timeline, [pct_highly_retained] * n_periods, '--b')
    ax.text(timeline[-1] - 8, pct_highly_retained + 3, f'{pct_highly_retained:.0f}% forever retained', color='b')

ax.set_ylim([0, 110])
ax.set_ylabel('% Retained')
ax.set_xlabel('Number of periods')
tab1.pyplot(fig)

with tab2:
    if df is not None:
        st.markdown(df.to_markdown())
    else:
        st.markdown('''
        Use the menu on the left to upload a file with your retention data.
        
        [Here you have an example](https://docs.google.com/spreadsheets/d/14RXaS_eaghqtPXA_6_Zf1oEoN-WRwbT7XW4iODcxK88/edit#gid=0) of the format that the data must have. Feel free
        to download the example and upload it in the app.
        ''')

tab3.markdown(explanation)

tab1.markdown(f"Expected state after {n_periods} periods:")

if survival_model == 'Harmonic':
    customers = expected_customers_harmonic(n_periods, avg_new_customers)
    growth = expected_customer_growth_harmonic(n_periods, avg_new_customers)
else:
    customers = expected_customers_weibull_mixture(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained / 100)
    growth = expected_customer_growth_weibull_mixture(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained / 100)

churned = avg_new_customers - growth
churn_rate = 100*churned/customers

col1, col2, col3 = tab1.columns(3)
col1.metric("Number of customers", f"{customers[-1]:.0f}")
col2.metric("Net new customers", f"{growth[-1]:.0f}")
col3.metric("Churn rate", f"{churn_rate[-1]:.0f}%")

tab11, tab12, tab13 = tab1.tabs(['Number of customers', 'Customer growth', 'Churn rate'])

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(timeline, customers)
ax.set_ylabel('Number of customers')
ax.set_xlabel('Number of periods')
tab11.pyplot(fig)
print(customers)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(timeline, growth)
ax.set_ylabel('Net customer growth')
ax.set_xlabel('Number of periods')
ax.set_ylim([0, avg_new_customers])
tab12.pyplot(fig)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(timeline, churn_rate)
ax.set_ylabel('Churn rate (%)')
ax.set_xlabel('Number of periods')
ax.set_ylim([0, np.nanmax(churn_rate)])
tab13.pyplot(fig)