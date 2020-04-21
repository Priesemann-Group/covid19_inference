import covid19_inference as cov19
import pymc3 as pm
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import theano
import theano.tensor as tt

def plot_compact(model, trace, new_cases_obs, date_data_begin, diff_data_sim=16, start_date_plot=None, end_date_plot=None, week_interval=None, colors = ('tab:blue', 'tab:orange'), savefig=None):

	def conv_time_to_mpl_dates(arr):
		return matplotlib.dates.date2num([datetime.timedelta(days=float(date)) + date_begin_sim for date in arr])

	#Sets up figure geometry
	len_A = 4
	len_B = 3

	fig = plt.figure(figsize=(12,8))
	gs = fig.add_gridspec(3*len_A,2*len_A+len_B)    

	ax_lambda = fig.add_subplot(gs[0:len_A,0:2*len_A])
	ax_cases = fig.add_subplot(gs[len_A:3*len_A,0:2*len_A])
	ax_I_begin = fig.add_subplot(gs[0:len_B,2*len_A:2*len_A+len_B])
	ax_delay = fig.add_subplot(gs[len_B:2*len_B,2*len_A:2*len_A+len_B])
	ax_mu = fig.add_subplot(gs[2*len_B:3*len_B,2*len_A:2*len_A+len_B])
	ax_sigma_obs = fig.add_subplot(gs[3*len_B:4*len_B,2*len_A:2*len_A+len_B])

	#Sets variables
	new_cases_sim = trace['new_cases']

	date_begin_sim = date_data_begin - datetime.timedelta(days = diff_data_sim)

	len_sim = trace['lambda_t'].shape[1]
	if start_date_plot is None:
		start_date_plot = date_begin_sim + datetime.timedelta(days=diff_data_sim)
	if end_date_plot is None:
		end_date_plot = date_begin_sim + datetime.timedelta(days=len_sim)

	num_days_data = len(new_cases_obs)
	diff_to_0 = num_days_data + diff_data_sim
	date_data_end = date_begin_sim + datetime.timedelta(days=diff_data_sim + num_days_data)
	num_days_future = (end_date_plot - date_data_end).days
	start_date_mpl, end_date_mpl = matplotlib.dates.date2num([start_date_plot, end_date_plot])
	week_inter_left = int(np.ceil(num_days_data/7/5))
	week_inter_right = int(np.ceil((end_date_mpl - start_date_mpl)/7/6))

	#Plots lambda
	ax = ax_lambda

	new_cases_sim = trace['new_cases']
	len_sim = trace['lambda_t'].shape[1]    

	time = np.arange(-diff_to_0 , -diff_to_0 + len_sim )
	lambda_t = trace['lambda_t'][:, :]
	μ = trace['mu'][:, None]
	mpl_dates = conv_time_to_mpl_dates(time) + diff_data_sim + num_days_data

	ax.plot(mpl_dates, np.median(lambda_t - μ, axis=0), color=colors[1], linewidth=2, label=r'$\lambda^*$')
	ax.fill_between(mpl_dates, np.percentile(lambda_t - μ, q=2.5, axis=0), np.percentile(lambda_t - μ, q=97.5, axis=0),
					alpha=0.15,
					color=colors[1])

	#ax.set_ylabel('effective\ngrowth rate $\lambda_t^*$')

	#ax.set_ylim(-0.15, 0.45)

	ylims = ax.get_ylim()
	ax.hlines(0, start_date_mpl, end_date_mpl, linestyles=':')
	delay = matplotlib.dates.date2num(date_data_end) - np.percentile(trace['delay'], q=75)
	ax.vlines(delay, ylims[0], ylims[1], linestyles='-', colors=['tab:red'])
	ax.set_ylim(*ylims)
	ax.text(delay + 0.5, ylims[1] - 0.04*np.diff(ylims), 'unconstrained because\nof reporting delay', color='tab:red', verticalalignment='top')
	ax.text(delay - 0.5, ylims[1] - 0.04*np.diff(ylims), 'constrained\nby data', color='tab:red', horizontalalignment='right',
			verticalalignment='top')
	ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval = week_inter_right, byweekday=matplotlib.dates.SU))
	ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
	ax.set_xlim(start_date_mpl, end_date_mpl)


	#Plots new cases
	ax = ax_cases

	time1 = np.arange(-len(new_cases_obs) , 0)
	mpl_dates = conv_time_to_mpl_dates(time1) + diff_data_sim + num_days_data
	ax.plot(mpl_dates, new_cases_obs, 'd', label='Data', markersize=4, color=colors[0],
			zorder=5)

	new_cases_past = new_cases_sim[:, :num_days_data]
	ax.plot(mpl_dates, np.median(new_cases_past, axis=0), '--', color=colors[1], linewidth=1.5, label='Fit with 95% CI')
	percentiles = np.percentile(new_cases_past, q=2.5, axis=0), np.percentile(new_cases_past, q=97.5, axis=0)
	ax.fill_between(mpl_dates, percentiles[0], percentiles[1], alpha=0.2, color=colors[1])

	time2 = np.arange(0, num_days_future)
	mpl_dates_fut = conv_time_to_mpl_dates(time2) + diff_data_sim + num_days_data
	cases_future = new_cases_sim[:, num_days_data:num_days_data+num_days_future].T
	median_cases = np.median(cases_future, axis=-1)
	percentiles = (
		np.percentile(cases_future, q=2.5, axis=-1),
		np.percentile(cases_future, q=97.5, axis=-1),
	)
	ax.plot(mpl_dates_fut, median_cases, color=colors[1], linewidth=3, label='forecast with 75% and 95% CI')
	ax.fill_between(mpl_dates_fut, percentiles[0], percentiles[1], alpha=0.1, color=colors[1])
	ax.fill_between(mpl_dates_fut, np.percentile(cases_future, q=12.5, axis=-1),
					np.percentile(cases_future, q=87.5, axis=-1),
					alpha=0.2, color=colors[1])

	ax.set_xlabel('Date')
	ax.set_ylabel(f'New confirmed cases')
	ax.legend(loc='upper left')
	func_format = lambda num, _: "${:.0f}\,$k".format(num / 1_000)
	ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_format))
	ax.set_xlim(start_date_mpl, end_date_mpl)
	ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval = week_inter_right, byweekday=matplotlib.dates.SU))
	ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))

	#Plots the parameters
	cov19.plotting.plot_hist(model, trace, ax_I_begin, 'I_begin', colors=("#708090", "tab:orange"), bins=50)
	cov19.plotting.plot_hist(model, trace, ax_delay, 'delay', colors=("#708090", "tab:orange"), bins=50)
	cov19.plotting.plot_hist(model, trace, ax_mu, 'mu', colors=("#708090", "tab:orange"), bins=50)
	cov19.plotting.plot_hist(model, trace, ax_sigma_obs, 'sigma_obs', colors=("#708090", "tab:orange"), bins=50)

	plt.tight_layout()

	if savefig is not None:
		fig.savefig(savefig + '.png', dpi=300)

def analyze_country(country, date_data_begin, date_data_end, N, mobility_type = 'transit_stations', mobility_source = 'google',lambda_0 = 0.4, lambda_min=0.1, num_days_forecast = 10, diff_data_sim=16):


	#Parses strings
	date_data_begin = datetime.datetime.fromisoformat(date_data_begin)
	date_data_end = datetime.datetime.fromisoformat(date_data_end)

	#Gets data
	data = cov19.data_retrieval.JHU()
	data.download_confirmed()
	new_cases = data.get_confirmed('Germany').diff()[date_data_begin:date_data_end].to_numpy()

	country_report = parse_names(country, mobility_source)
	if mobility_source == 'apple':
		mobility = get_mobility_reports_apple(country_report,[mobility_type])
	elif mobility_source == 'google':
		mobility = get_mobility_reports_google(country_report,[mobility_type])
	else:
		ValueError('Invalid mobility_source')

	# if date_data_end > mobility.index.max():
	# 	ValueError('date_data_end later than {}'.format(str(mobility.index.max())))

	#change_points = mobility_to_changepoints(mobility[date_data_begin:date_data_end], lambda_0 = lambda_0, lambda_min=lambda_min)



	params_model = dict(new_cases_obs = new_cases,
					date_begin_data = date_data_begin,
					num_days_forecast = num_days_forecast,
					diff_data_sim = diff_data_sim,
					N_population = N) 

	with cov19.Cov19_Model(**params_model) as model:
		#lambda_t_log = cov19.lambda_t_with_sigmoids(pr_median_lambda_0 = lambda_0,
		#											change_points_list = change_points)

		ts = mobility[date_data_begin:date_data_end].to_numpy().flatten()
		ts = np.concatenate((ts,ts[-1]*np.ones(num_days_forecast+1)))

		lambda_t_log = lambda_t_single(ts)
		
		new_I_t = cov19.SIR(lambda_t_log, pr_median_mu=1/8)
		
		new_cases_inferred_raw = cov19.delay_cases(new_I_t, pr_median_delay=10, 
												   pr_median_scale_delay=0.3)
		
		
		cov19.student_t_likelihood(new_cases_inferred_raw)


	trace = pm.sample(model=model, init='advi', tune=500, draws=500, cores=6)

	return trace, model

	str_save = '{}_{}_{}'.format(country, mobility_source, mobility_type)
	#plot_compact(model, trace, new_cases, date_data_begin, diff_data_sim, savefig=str_save)
	
def mobility_to_changepoints(mobility, lambda_0, lambda_min):

	# default_priors_change_points = dict(
	# 	pr_median_lambda=0.4,
	# 	pr_sigma_lambda=0.5,
	# 	pr_sigma_date_begin_transient=3,
	# 	pr_median_transient_len=3,
	# 	pr_sigma_transient_len=0.3,
	# 	pr_mean_date_begin_transient=None,
	# )

	#Rescales values
	mobility_min = mobility.min()[0]
	mobility_max = mobility.max()[0]
	mobility = lambda_min + (lambda_0-lambda_min)*(mobility-mobility_min)/(mobility_max-mobility_min)

	mobility.index.name = 'date'
	mobility_df = mobility.reset_index()
	change_points = []
	for i in range(len(mobility)):
		change_point = dict(
		pr_median_lambda=mobility.iloc[i][0],
		pr_sigma_lambda=1,
		pr_sigma_date_transient=1,
		pr_median_transient_len=1,
		pr_sigma_transient_len=1,
		pr_mean_date_transient=mobility.index[i],
		)
 
		change_points.append(change_point)

	return change_points

def parse_names(country, source):

	if source == 'google':
		parse_names = {'Korea, South': 'South Korea'}
	elif source == 'apple':
		parse_names = {'US': 'United States', 'Korea, South':'Republic of Korea', 'United Kingdom':'UK'}
	else:
		ValueError('Invalid source. Valid options: "google", "apple"')

	if country in parse_names.keys():
		return parse_names[country]
	else:
		return country

def analyze_all(end_date, mobility_source):

	#Name convention: JHU
	country_list  = ['US','Spain','Italy','France','Germany','United Kingdom','Turkey','Belgium','Canada','Netherlands','Switzerland','Russia','Portugal','Austria','Ireland','Israel','Sweden', 'Korea, South']

	country_pop  = {'US':328e6,'Spain':47e6,'Italy':60.3e6,'France':67e6,'Germany':83e6,'United Kingdom':66.6e6,'Turkey':82e6,'Belgium':11.5e6,'Canada':37.6e6,'Netherlands':17.3e6,'Switzerland':8.6e6,'Russia':144e6,'Portugal':10.3e6,'Austria':8.9e6,'Ireland':4.9e6,'Israel':8.9e6,'Sweden':10.2e6, 'Korea, South':51.6e6}

	for country in country_list:
		auto_analysis.analyze_country(country,'2020-03-01',end_date,N=country_pop[country], mobility_source='apple', mobility_type='transit')

def get_mobility_reports_apple(value, transportation_list, path_data = 'data/applemobilitytrends-2020-04-13.csv'):

    if not all(elem in ['walking', 'driving', 'transit']  for elem in transportation_list):
        raise ValueError('transportation_type contains elements outside of ["walking", "driving", "transit"]')

    # if transportation_type not in ['walking', 'driving', 'transit']:
    #     raise ValueError('Invalid value. Valid options: "walking", "driving", "transit"')

    df = pd.read_csv(path_data)

    series_list = []
    for transport in transportation_list:
        series = df[(df['region']==value) & (df['transportation_type']==transport)].iloc[0][3:].rename(transport)
        series_list.append(series/100)

    df2 = pd.concat(series_list,axis=1)

    df2.index = df2.index.map(datetime.datetime.fromisoformat)

    return df2
    
def get_mobility_reports_google(region, field_list, subregion=False):

    valid_fields = ['retail_and_recreation','grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces','residential']

    if not all(elem in valid_fields  for elem in field_list):
        raise ValueError('field_list contains invalid elements')


    url = 'https://raw.githubusercontent.com/vitorbaptista/google-covid19-mobility-reports/master/data/processed/mobility_reports.csv'
    df = pd.read_csv(url)

    if subregion is not False:
        series_df = df[(df['region']==region) & (df['subregion'] == subregion)]
    else:
        series_df = df[(df['region']==region) & (df['subregion'].isnull())]

    series_df = series_df.set_index('updated_at')[field_list]
    series_df.index.name = 'date'
    series_df.index = series_df.index.map(datetime.datetime.fromisoformat)
    series_df = series_df + 1

    return series_df


def lambda_t_single(ts):

	cov19.Cov19_Model.get_context()

	a = pm.Uniform(name='a', lower=0, upper=1)
	b = pm.Uniform(name='b', lower=-1, upper=1)

	ts = tt._shared(ts.astype("float64"))

	lambda_t_log = tt.log(a*ts + b)

	pm.Deterministic('lambda_t', a*ts + b)

	return lambda_t_log