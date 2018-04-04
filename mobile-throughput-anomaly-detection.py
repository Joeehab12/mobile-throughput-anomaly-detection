import warnings
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv('interpolated-detailed.csv')
df.set_index('datetime')
subset = df[(df['datetime'] > "2016-08-14 00:00:00") & (df['datetime'] < "2016-09-14 00:00:00")]
subset = subset.iloc[:, 0:2]
subset.set_index('datetime')

#test
#print(subset['datetime'])
plt.style.use('ggplot')

sum = 0

list2 = []
for i in range (0,24):
    if i < 10:
        i = "0" + str(i)
    for j in np.arange(15,60,15):
        if j < 10:
            j = "0" + str(j)
        print('2016-08-14 ' + str(i) + ':' + str(j) + ':00')
        xy = subset[(subset['datetime'] == '2016-08-14 ' + str(i) + ':' + str(j) + ':00')]
        xy = xy.iloc[:, 0:1]
        sum = sum + float(xy.value)
    list2.append(sum/5)
    sum = 0
# test
index = [i for i in (0,len(list2))]
hourly = pd.DataFrame(list2)

hourly.plot(label='hourly')
plt.title('Hourly Data')
plt.show()
avg = (sum)/96

print(avg)
#subset = subset.set_index('datetime')



a = [1,2,3,4,5]
b = [6,7,8,9,10]

subset2 = subset
print(len(subset2))

x = [i for i in range(1,len(subset2) + 1)]
subset2.index = x
subset2 = subset2.iloc[:, 0:1]
#
plt.scatter(subset2.index[1:96],subset2[1:96])
plt.show()

subset['value'] = np.log(subset['value'])
subset.columns = ['y','ds']
m = Prophet()
m.fit(subset)
future = m.make_future_dataframe(periods=4)

forecast = m.predict(future)
m.plot(forecast)
m.plot_components(forecast)
plt.title('Prophet Forecast')
#plt.xlim(0,100)
plt.show()



subset = subset.set_index('ds')

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
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

mod = sm.tsa.ARIMA(subset,order = (1,0,2))

results = mod.fit()

ax = subset['2016-08-31 00:00:00':'2016-09-03 00:15:00']
ax_plot = ax.plot(label='observed',scalex = False)
results.fittedvalues['2016-09-02 00:00:00':'2016-09-05 00:00:00'].plot(label = "One-step ahead Forecast" ,alpha = 0.7)

root_mean_squared_error = len(results.fittedvalues['2016-09-02 00:00:00':'2016-09-05 00:00:00'])
ax_len =  len(ax['2016-08-31 00:00:00':'2017-09-01 00:00:00'])


results.fittedvalues.index = [i for i in range(0,len(results.fittedvalues))]
y = results.fittedvalues
y.columns = ['value']
y.index = [i for i in range(0,len(y))]
y = np.asarray(y,dtype='float')

x = [i for i in range(0,len(ax))]
ax.index = x
ax = np.asarray(ax,dtype='float')
rmse = np.sqrt(((y[0:96] - ax[0:96])**2))

ax_plot.set_xlabel('Datetime')
ax_plot.set_ylabel('Mobile Throughput')
plt.legend()
plt.xlim(0,200)
plt.show()

rmse_df = pd.DataFrame(rmse)

print('standard deviation = {} ,mean = {}'.format(np.std(rmse_df),np.mean(rmse_df)))
rmse_df[0].plot()   #plot root mean square error for first day
plt.title('Root Mean Squared Error')
plt.show()



test = sm.tsa.seasonal_decompose(subset.values,freq= 94,model = "additive")

test.plot()
plt.show()