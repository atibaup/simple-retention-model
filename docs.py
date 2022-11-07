explanation=r'''
This app uses a retention model corresponding to a mixture of a Weibull 
and an infinite survival model, where the share of customers being retained 
at time $k$ is given by the expression:

$$ s(k) = \exp\left(-\left(\frac{k}{\lambda} \right)^\rho\right) (1-p) + p $$

with parameters:
- $p$: the share of customers who are forever retained,
- $\lambda$: roughly controls the average lifetime (time to churn) of a customer,
- $\rho$: the rate at which the rate of churn increases or decreases as time goes by. 

Note that for $\rho=1$ and $p=0$ this model corresponds to an exponential (or geometric) survival model:

$$ s(k) = \exp\left(-\frac{k}{\lambda} \right) $$

where a proportion of $\exp\left(-\frac{1}{\lambda}\right)$ customers churn at every period.

For $p=1$ it boils down to a classical Weibull survival model.

Given the above retention model, and assuming the number of new customers $N$
per period remains constant, the tool displays the number of customers $C(t)$,
 
$$
C(t) = \sum_{k=0}^t s(k) N
$$

the customer growth $\Delta C(t)$ and the churn rate
as a function of the number of periods since the start.
'''