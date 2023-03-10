#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

__all__ = [
    "normalised_excess_variance"
]


from scipy.stats import chi2, norm
import numpy as np


def normalised_excess_variance(
        lc_table, 
        alpha_level_sigmas, 
        percent_sys_error=3, 
        chi_squared_test=False
):
    """
    Compute the p-value of the Normalised_excess_variance test or a simple
    chi-squared test against the null hypothesis of a constant flux. 
    Both statistical and systematic errors are considered. 
    
    Parameters
    ----------
    lc_table:
        Table with fluxes
    
    alpha_level_sigmas:
        The alpha level value in sigmas.

    percent_sys_error: int, float
        Percentage of the systematic error of the mean fluxes.
        Default value of 3%.

    chi_squared_test: bool
        If true, a simple the simple chi-squared test is computed.
        Else, the normalised excess variance is computed.
        Default value is False.
        
    Returns
    -------
    p_value: float
        p-value against the null hypothesis.
    
    chi2_value: float
        The chi-squared value given the data.
        
    critical: float
        chi-squared critical value to reject/accept the null hypothesis.
    """
    
    flux = []
    flux_err = []
    time = []
    for i in range(len(lc_table["flux"])):
        if lc_table["is_ul"][i] is False:
            flux.append(lc_table["flux"][i])
            flux_err.append(lc_table["flux_err"][i])
            time.append(lc_table["time_max"][i]-lc_table["time_min"][i])
            
    flux = np.array(flux)
    flux_err = np.array(flux_err)
    time = np.array(time)
    sist_err = percent_sys_error*flux.mean()
    
    dof = len(flux) - 1
    
    sigma_prob = 2 * norm.cdf(alpha_level_sigmas) - 1
    alpha_level = 1 - sigma_prob
    
    critical = chi2.ppf(1-alpha_level, dof)
    
    if chi_squared_test:
        chi2_value = ((flux-(flux.mean())**2)/(flux_err**2+sist_err**2)).sum()
    else:
        chi2_value = ((time*(flux-flux.mean())**2)/((flux_err**2+sist_err**2)*time.sum())).sum()
    
    p_value = chi2.sf(chi2_value, df=dof)
    if p_value >= alpha_level:
        print("Null hypothesis is accepted")
    else:
        print("Null hypothesis is rejected, variable source")

    return p_value, chi2_value, critical
