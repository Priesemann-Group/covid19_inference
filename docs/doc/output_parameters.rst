Variables saved in the trace
============================

The trace by default contains the following parameters in the
SIR/SEIR hierarchical model. XXX denotes a number.

.. list-table::
    :widths: 25 25 100
    :header-rows: 1

    * - Name in trace
      - Dimensions
      - Created by function

    * - lambda_XXX_L1
      - samples
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - lambda_XXX_L2
      - samples x regions
      - lambda_t_with_sigmoids/make_change_point_RVs


    * - sigma_lambda_XXX_L2
      - samples
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - transient_day_XXX_L1
      - samples
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - transient_day_XXX_L2
      - samples x regions
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - sigma_transient_day_XXX_L2
      - samples
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - transient_len_XXX_L1
      - samples
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - transient_len_XXX_L2
      - samples x regions
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - sigma_transient_len_XXX_L2
      - samples
      - lambda_t_with_sigmoids/make_change_point_RVs

    * - delay_L1
      - samples
      - delay_cases

    * - delay_L2
      - samples x regions
      - delay_cases

    * - sigma_delay_L2
      - samples
      - delay_cases

    * - weekend_factor_L1
      - samples
      - week_modulation

    * - weekend_factor_L2
      - samples x regions
      - week_modulation

    * - sigma_weekend_factor_L2
      - samples
      - week_modulation

    * - offset_modulation
      - samples
      - week_modulation

    * - new_cases_raw
      - samples x time x regions
      - week_modulation

    * - mu
      - samples
      - SIR/SEIR

    * - I_begin
      - samples x regions
      - SIR/SEIR

    * - new_cases
      - samples x time x regions
      - SIR/SEIR

    * - sigma_obs
      - samples x regions
      - SIR/SEIR

    * - new_E_begin
      - samples x 11 x regions
      - SEIR

    * - median_incubation_L1
      - samples
      - SEIR

    * - median_incubation_L2
      - samples x regions
      - SEIR

    * - sigma_median_incubation_L2
      - samples
      - SEIR

For the non-hierchical model, variables with _L2 suffixes are missing, and _L1 suffixes
are removed from the name.


