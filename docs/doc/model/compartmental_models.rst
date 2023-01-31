Compartmental models
====================

.. contents:: Table of Contents
	:depth: 2


SIR --- susceptible-infected-recovered
--------------------------------------

.. autofunction:: covid19_inference.model.compartmental_models.SIR

.. admonition:: More Details

    .. math::

        I_{new}(t) &= \lambda_t I(t-1)  \frac{S(t-1)}{N}   \\
        S(t) &= S(t-1) - I_{new}(t)  \\
        I(t) &= I(t-1) + I_{new}(t) - \mu  I(t)

    The prior distributions of the recovery rate :math:`\mu`
    and :math:`I(0)` are set to

    .. math::

        \mu &\sim \text{LogNormal}\left[
                \log(\text{pr_median_mu}), \text{pr_sigma_mu}
            \right] \\
        I(0) &\sim \text{HalfCauchy}\left[
                \text{pr_beta_I_begin}
            \right]


SEIR-like ---  susceptible-exposed-infected-recovered
-----------------------------------------------------

.. autofunction:: covid19_inference.model.compartmental_models.SEIR

.. admonition:: More Details

    .. math::

        E_{\text{new}}(t) &= \lambda_t I(t-1) \frac{S(t)}{N}   \\
        S(t) &= S(t-1) - E_{\text{new}}(t)  \\
        I_\text{new}(t) &= \sum_{k=1}^{10} \beta(k) E_{\text{new}}(t-k)   \\
        I(t) &= I(t-1) + I_{\text{new}}(t) - \mu  I(t) \\
        \beta(k) & = P(k) \sim \text{LogNormal}\left[
                \log(d_{\text{incubation}}), \text{sigma_incubation}
            \right]

    The recovery rate :math:`\mu` and the incubation period is the same for all regions and follow respectively:

    .. math::

        P(\mu) &\sim \text{LogNormal}\left[
                \text{log(pr_median_mu), pr_sigma_mu}
            \right] \\
        P(d_{\text{incubation}}) &\sim \text{Normal}\left[
                \text{pr_mean_median_incubation, pr_sigma_median_incubation}
            \right]

    The initial number of infected and newly exposed differ for each region and follow prior :class:`pymc.HalfCauchy` distributions:

    .. math::

        E(t)  &\sim \text{HalfCauchy}\left[
                \text{pr_beta_E_begin}
            \right] \:\: \text{for} \: t \in {-9, -8, ..., 0}\\
        I(0)  &\sim \text{HalfCauchy}\left[
                \text{pr_beta_I_begin}
            \right].


Kernelized Spread
-----------------

.. autofunction:: covid19_inference.model.compartmental_models.kernelized_spread


.. admonition:: Reference

    * .. [Nishiura2020]
        Nishiura, H.; Linton, N. M.; Akhmetzhanov, A. R.
        Serial Interval of Novel Coronavirus (COVID-19) Infections.
        Int. J. Infect. Dis. 2020, 93, 284–286. https://doi.org/10.1016/j.ijid.2020.02.060.
    * .. [Lauer2020]
        Lauer, S. A.; Grantz, K. H.; Bi, Q.; Jones, F. K.; Zheng, Q.; Meredith, H. R.; Azman, A. S.; Reich, N. G.; Lessler, J.
        The Incubation Period of Coronavirus Disease 2019 (COVID-19) From Publicly Reported Confirmed Cases: Estimation and Application.
        Ann Intern Med 2020. https://doi.org/10.7326/M20-0504.
