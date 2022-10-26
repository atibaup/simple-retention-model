import autograd.numpy as np
from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.utils.safe_exp import safe_exp
from lifelines import utils

import autograd.numpy as anp


class WeibullInfiniteMixtureFitter(KnownModelParametricUnivariateFitter):
    r"""
    cf: https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/fitters/weibull_fitter.py
    """

    lambda_: float
    rho_: float
    p_: float
    _fitted_parameter_names = ["lambda_", "rho_", "p_"]
    _compare_to_values = np.array([1.0, 1.0, 0.])
    _scipy_fit_options = {"ftol": 1e-14}

    def _create_initial_point(self, Ts, E, entry, weights):
        return np.array([utils.coalesce(*Ts).mean(), 1.0, 0.1])

    def _survival_function(self, params, times):
        lambda_, rho_, p_ = params
        return anp.exp(-np.power(times/lambda_, rho_))*(1-p_) + p_