import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from lifelines import WeibullFitter
from st_aggrid import AgGrid, GridOptionsBuilder

plt.style.use('dark_background')


def weibull_survival(timelines, lambda_, rho_):
    return np.exp(-np.power(timelines/lambda_, rho_))


def mixed_weibull_and_retained(timelines, lambda_, rho_, gamma):
    return gamma + (1-gamma)*weibull_survival(timelines, lambda_, rho_)


def expected_customers(n_periods, N_0, lambda_, rho_, gamma):
    retained = np.zeros(n_periods)
    periods = np.array([i for i in range(n_periods)])
    retention = mixed_weibull_and_retained(periods, lambda_, rho_, gamma)
    for n in range(1, n_periods):
        retained[n] = N_0 * sum(retention[:n])
    return retained


def expected_customer_growth(n_periods, N_0, lambda_, rho_, gamma):
    periods = np.array([i for i in range(n_periods)])
    retention = mixed_weibull_and_retained(periods, lambda_, rho_, gamma)
    return N_0 * retention


def convert_retention_cohorts_to_survival_data(retention_df):
    data_in_survival_format = []
    for date, row in retention_df.iterrows():
        n_cohort = row[0]
        n_cohort_churned = 0
        for i, n_churned in enumerate(row.diff().abs()):
            if i > 0:  # skip first col since it's always a NaN
                if not np.isnan(n_churned):
                    for j in range(int(n_churned)):
                        data_in_survival_format.append(
                            (date, i, True)
                        )
                    n_cohort_churned += n_churned
                else:
                    # first NaN found after the first means no data for that cohort and duration
                    break
        for j in range(int(n_cohort - n_cohort_churned)):
            data_in_survival_format.append((date, i, False))
    return pd.DataFrame(data_in_survival_format, columns=['date', 'duration', 'observed'])


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
    You can upload your retention data as a CSV or play with
     the parameters below to generate different scenarios:
     ''')

    uploaded_file = st.file_uploader(
        "Upload your retention data",
        help='The file must be CSV formatted with cohorts in its rows and periods from start as its columns')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=0, index_col=0)
        df.index.rename('cohort', inplace=True)
        df.rename(columns=lambda x: int(x), inplace=True)
        default_new_customers = round(df[0].mean())

        data_in_survival_format = convert_retention_cohorts_to_survival_data(df)
        aggregate = calculate_aggregate_retention(df)
        wf = WeibullFitter().fit(data_in_survival_format['duration'], data_in_survival_format['observed'])
        lambda_default = float(round(wf.lambda_, 2))
        rho_default = float(round(wf.rho_, 2))
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
    lambda_ = st.slider('Average life time', 0., 20., lambda_default, help='Larger values correspond to larger life times', step=1.)
    rho_ = st.select_slider('Churn rate deceleration/aceleration', options=sorted([0.01, 0.1, 0.5, 0.75, 1., 2., 5., 10., rho_default]), value=rho_default)
    pct_highly_retained = st.slider('% forever retained', 0., 100., pct_highly_retained_default, step=5.)

    agree = st.checkbox('Display % forever retained')

timeline = np.array([i for i in range(n_periods)])

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
        df_aggrid = df.reset_index().rename(columns=lambda x: str(x))
        builder = GridOptionsBuilder.from_dataframe(df_aggrid)
        builder.configure_auto_height(autoHeight=False)
        builder.configure_default_column(width=50, editable=False)
        builder.configure_column('cohort', width=100)
        go = builder.build()
        grid_return = AgGrid(df_aggrid, gridOptions=go)
    else:
        st.markdown('Use the menu on the left to upload a file with your retention data')

tab3.markdown('''
This app uses a retention model corresponding to a mixture of a Weibull 
and an infinite survival model, where the share of customers being retained 
at time $k$ is given by the expression:
''')
tab3.latex(r'''
\gamma(k) = \exp\left(-\left(\frac{k}{\lambda} \right)^\rho\right) (1-p) + p
''')
tab3.markdown('''
with parameters:
- $p$: the share of customers who are forever retained,
- $\lambda$: roughly controls the average lifetime (time to churn) of a customer,
- $\\rho$: the rate at which the rate of churn increases or decreases as time goes by. 

Note that for $\\rho=1$ and $p=0$ this would correspond to an exponential (or geometric) survival model:
''')
tab3.latex(r'''
\gamma(k) = \exp\left(-\frac{k}{\lambda} \right)
''')
tab3.markdown('''
where a proportion of $\\exp\\left(-\\frac{1}{\\lambda}\\right)$ customers churn at every period.

Given the above retention model, and assuming the number of new customers $N$
per period remains constant, the tool displays the number of customers $C(t)$,
the customer growth $\Delta C(t)$ and the churn rate
as a function of the number of periods since the start. 
''')
tab3.latex(r'''
C(t) = \sum_{k=0}^t \gamma(k)N
''')

tab1.markdown(f"Expected status after {n_periods} periods:")

customers = expected_customers(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained/100)
growth = expected_customer_growth(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained/100)
churned = avg_new_customers - expected_customer_growth(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained/100)
churn_rate = 100*churned/customers

col1, col2, col3 = tab1.columns(3)
col1.metric("Number of customers", f"{customers[-1]:.0f}")
col2.metric("Net new customers", f"{growth[-1]:.0f}")
col3.metric("Churn rate", f"{churn_rate[-1]:.0f}%")

tab11, tab12, tab13 = tab1.tabs(['Customers', 'Customer growth', 'Churn rate'])

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(timeline, customers)
ax.set_ylabel('Number of customers')
ax.set_xlabel('Number of periods')
tab11.pyplot(fig)

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