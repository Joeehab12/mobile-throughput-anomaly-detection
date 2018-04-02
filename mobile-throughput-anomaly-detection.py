import warnings
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv('interpolated-detailed.csv')
df.set_index('datetime')
subset = df[(df['datetime'] > "2016-08-14") & (df['datetime'] < "2016-12-31")]
subset = df.iloc[:, 0:2]
subset = subset.set_index("datetime")

plt.style.use('fivethirtyeight')


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore")  # specify to ignore warning messages

ARrange = [i for i in range(0,8)]
MArange = [i for i in range(0,8)]
Diffrange = (0,1)
def get_safe_RSS(series, fitted_values):
    """ Checks for missing indices in the fitted values before calculating RSS

    Missing indices are assigned as np.nan and then filled using neighboring points
    """
    fitted_values_copy = fitted_values  # original fit is left untouched
    missing_index = list(set(series.index).difference(set(fitted_values_copy.index)))
    return 1#sum((fitted_values_copy - series) ** 2)

def iterative_ARIMA_fit(series):
    ARIMA_fit_results = {}
    for AR in ARrange:
        for MA in MArange:
            for Diff in Diffrange:
                model = ARIMA(series, order=(AR, Diff, MA))
                fit_is_available = False
                results_ARIMA = None
                try:
                    results_ARIMA = model.fit(disp=-1, method='css')
                    fit_is_available = True
                except:
                    continue
                if fit_is_available:
                    safe_RSS = get_safe_RSS(series, results_ARIMA.fittedvalues)
                    ARIMA_fit_results['%d-%d-%d' % (AR, Diff, MA)] = [safe_RSS, results_ARIMA]
    return ARIMA_fit_results


#test22 = iterative_ARIMA_fit(subset)







# mod = sm.tsa.statespace.SARIMAX(subset,
#                                 order=(1,0,7),
#                                 seasonal_order=(0, 0, 0, 12),
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False)

mod = sm.tsa.ARIMA(subset,order = (7,0,2))

results = mod.fit()

ax = subset['2017-01-31 00:00:00':'2017-02-01 00:00:00'].plot(label='observed')
results.fittedvalues['2017-02-01 00:00:00':'2017-02-05 00:00:00'].plot(label = "One-step ahead Forecast" ,alpha = 0.7)

ax.set_xlabel('Datetime')
ax.set_ylabel('Mobile Throughput')
plt.legend()
plt.show()



results.plot_predict(start = pd.to_datetime("2017-02-01 23:45:00"),end = pd.to_datetime("2017-02-05 23:45:00"))
plt.show()

#test = sm.tsa.seasonal_decompose(subset.value,freq= 94,model = "additive")
