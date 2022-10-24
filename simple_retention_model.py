import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
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
    return np.floor(retained)


def expected_customer_growth(n_periods, N_0, lambda_, rho_, gamma):
    periods = np.array([i for i in range(n_periods)])
    retention = mixed_weibull_and_retained(periods, lambda_, rho_, gamma)
    return np.floor(N_0 * retention)


with st.sidebar:
    n_periods = st.number_input('Number of periods', value=12, min_value=6, max_value=60)
    avg_new_customers = st.number_input("New customers / period", min_value=10, value=100, step=100)

    st.subheader('Retention parameters')
    lambda_ = st.slider('Average life time', 0., 20., 5., help='Larger values correspond to larger life times', step=1.)
    rho_ = st.select_slider('Churn rate deceleration/aceleration', options=[0.01, 0.1, 0.5, 0.75, 1., 2., 5., 10.], value=1.)
    pct_highly_retained = st.slider('% forever retained', 0., 100., 30., step=5.)

    agree = st.checkbox('Display % forever retained')

timeline = np.array([i for i in range(n_periods)])

retention_profile = mixed_weibull_and_retained(timeline, lambda_, rho_, pct_highly_retained/100)

st.header("Simplified retention model")

tab1, tab2 = st.tabs(['Retention profile', 'Explanation'])

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(timeline, 100*retention_profile)
if agree:
    ax.plot(timeline, [pct_highly_retained] * n_periods, '--b')
    ax.text(timeline[0], pct_highly_retained + 2, f'{pct_highly_retained}% forever retained', color='b')
ax.set_ylim([0, 100])
ax.set_ylabel('% Retained')
ax.set_xlabel('Number of periods')
tab1.pyplot(fig)

tab2.text('''
This tool generates the retention curve corresponding to a Weibull survival model,
where the probability of a customer being retained after k or more periods is given by the
expression:
''')
tab2.latex(r'''
\exp\left(-\left(\frac{k}{\lambda} \right)^\rho\right)
''')
tab2.text('''
with parameters \lambda and \\rho roughly corresponding to the average lifetime 
(time to churn) of a customer and to the rate at which the rate of churn increases
 or decreases as time goes by, respectively.
 
Given that retention curve, and assuming the number of new customers
per period remains constant, the tool displays the number of customers,
the customer growth and the churn rate
as a function of the number of periods since the start:
''')
tab2.latex(r'''
\exp\left(-\left(\frac{k}{\lambda} \right)^\rho\right)
''')

tab1.subheader(f"Expected status after {n_periods} periods:")

customers = expected_customers(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained/100)
growth = expected_customer_growth(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained/100)
churned = avg_new_customers - expected_customer_growth(n_periods, avg_new_customers, lambda_, rho_, pct_highly_retained/100)
churn_rate = 100*churned/customers

col1, col2, col3 = tab1.columns(3)
col1.metric("Number of customers", f"{customers[-1]:.0f}")
col2.metric("Net new customers", f"{growth[-1]:.0f}")
col3.metric("Churn rate", f"{churn_rate[-1]:.2f}%")

tab11, tab12, tab13 = tab1.tabs(['Customers', 'Customer growth', 'Churn rate'])

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(customers)
ax.set_ylabel('Number of customers')
ax.set_xlabel('Number of periods')
tab11.pyplot(fig)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(growth)
ax.set_ylabel('Net customer growth')
ax.set_xlabel('Number of periods')
tab12.pyplot(fig)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(churn_rate)
ax.set_ylabel('Churn rate (%)')
ax.set_xlabel('Number of periods')
tab13.pyplot(fig)