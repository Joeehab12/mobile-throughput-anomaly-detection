import warnings
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import types
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
plt.close("all")
df = pd.read_csv('interpolated-detailed.csv')
#df.set_index('datetime')
subset = df[(df['datetime'] > "2016-08-14 00:00:00") & (df['datetime'] < "2016-08-15 00:00:00") ]
#subset3 = subset[subset['wday']== 'Sunday']
subset = subset.iloc[:, 0:2]
#subset['datetime'] = pd.to_datetime(subset['datetime']).dt.strftime('%Y/%m/%d %H:%M')


subset_log = np.log(subset['value'])
moving_avg = pd.rolling_mean(subset_log,2)
# plt.plot(subset_log)
# plt.plot(moving_avg, color='red')
# plt.show()

diff = subset_log - moving_avg


diff.index = pd.date_range(start = pd.to_datetime('2016-08-14 00:00:00'),periods=95, freq='15min')
diff['2016-08-14 00:00:00'] = 0
# columnsTitles=["datetime","value"]
# subset=subset.reindex(columns=columnsTitles)
# subset.to_csv('exported.csv',index = False)

#subset = subset.set_index('datetime')

#test
#print(subset['datetime'])
plt.style.use('ggplot')
#
# sum = 0
#
# list2 = []
# for i in range (0,24):
#     if i < 10:
#         i = "0" + str(i)
#     for j in np.arange(15,60,15):
#         if j < 10:
#             j = "0" + str(j)
#         print('2016-08-14 ' + str(i) + ':' + str(j) + ':00')
#         xy = subset[(subset['datetime'] == '2016-08-14 ' + str(i) + ':' + str(j) + ':00')]
#         xy = xy.iloc[:, 0:1]
#         sum = sum + float(xy.value)
#     list2.append(sum/5)
#     sum = 0
# # test
# index = [i for i in (0,len(list2))]
# hourly = pd.DataFrame(list2)
#
# hourly.plot(label='hourly')
# plt.title('Hourly Data')
# plt.show()
# avg = (sum)/96
#
# print(avg)

# subset = subset.set_index('datetime')
#
#
#
# a = [1,2,3,4,5]
# b = [6,7,8,9,10]
#
# subset2 = subset
# print(len(subset2))
#
# x = [i for i in range(1,len(subset2) + 1)]
# subset2.index = x
# subset2 = subset2.iloc[:, 0:1]
#
# plt.scatter(subset2.index[1:96],subset2[1:96])
# plt.show()
#
# #subset['value'] = np.log(subset['value'])
# subset.columns = ['y','ds']
# # m = Prophet()
# # m.fit(subset)
# # future = m.make_future_dataframe(periods=1)
# # #
# # # forecast2 = m.predict(future)
# # # m.plot(forecast2)
# # # plt.title('Prophet Forecast')
# # # plt.show()
# # # m.plot_components(forecast2)
# # # #plt.xlim(0,100)
# # # plt.title('Prophet Forecast Components')
# # # plt.show()
# #
#
#
# subset = subset.set_index('ds')
#
# test = sm.tsa.seasonal_decompose(subset.values[0:96*12],freq= 94,model = "additive")
#
# test.plot()
# plt.show()
#
# # Define the p, d and q parameters to take any value between 0 and 2
# p = d = q = range(0, 2)
#
# # Generate all different combinations of p, q and q triplets
# pdq = list(itertools.product(p, d, q))
#
# # Generate all different combinations of seasonal p, q and q triplets
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#
# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#
warnings.filterwarnings("ignore")  # specify to ignore warning messages

ARrange = [i for i in range(0,8)]
MArange = [i for i in range(0,8)]
Diffrange = [i for i in range(0,2)]
def get_safe_RSS(series, fitted_values):
    """ Checks for missing indices in the fitted values before calculating RSS

    Missing indices are assigned as np.nan and then filled using neighboring points
    """
    fitted_values_copy = fitted_values  # original fit is left untouched
    missing_index = list(set(series.index).difference(set(fitted_values_copy.index)))
    return sum((fitted_values_copy - series) ** 2)

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
                except Exception as e:
                    print(str(e))
                    continue
                if fit_is_available:
                    safe_RSS = get_safe_RSS(series, results_ARIMA.fittedvalues)
                    ARIMA_fit_results['%d-%d-%d' % (AR, Diff, MA)] = [safe_RSS, results_ARIMA]
    return ARIMA_fit_results


#test22 = iterative_ARIMA_fit(diff)

# min = 10000
# min_x = 0
# for x in test22:
#     if test22[x][1].aic <= min:
#         min = test22[x][1].aic
#         min_x = x
#     print('{} x {}'.format(x,test22[x][1].aic))
# print('min AIC = ,parameters = ',min,min_x)
# #
# #mod = sm.tsa.statespace.SARIMAX(subset,
# #                                 order=(7,0,2),
# #                                 seasonal_order=(0, 0, 0, 12),
# #                                 enforce_stationarity=False,
# #                                 enforce_invertibility=False)
#
# mod = sm.tsa.ARIMA(subset,order = (4,0,6))
#
# results = mod.fit(disp=False)
#
# predicted = results.fittedvalues
# arima_results = results.forecast(steps = 96*2,alpha = 0.05)
#
#
# results.plot_predict(start = pd.to_datetime('2016-12-30 00:00:00'),end = pd.to_datetime('2017-01-02 00:00:00'))
# plt.show()
# forecast, stderr, conf = arima_results
#
# forecast_plot = pd.DataFrame(forecast)
# forecast_plot.plot(label = "forecasted value")
# plt.show()
# # predicted[0:96].plot()
# # plt.show()
#
#
# ax = subset['2016-08-13 00:00:00':'2016-09-03 00:15:00']
# ax_plot = ax.plot(label='observed',scalex = False)
#
# # # ax_plot.fill_between(conf.index,
# # #                 conf.iloc[:, 0],
# #                 conf.iloc[:, 1], color='k', alpha=.2)
#
# results.fittedvalues['2016-09-03 00:15:00':'2016-09-05 00:00:00'].plot(label = "One-step ahead Forecast" ,alpha = 0.7)
#
# root_mean_squared_error = len(results.fittedvalues['2016-09-03 00:15:00':'2016-09-05 00:00:00'])
# ax_len =  len(ax['2016-08-31 00:00:00':'2017-09-01 00:00:00'])
#
#
# results.fittedvalues.index = [i for i in range(0,len(results.fittedvalues))]
# y = results.fittedvalues
# y.columns = ['value']
# y.index = [i for i in range(0,len(y))]
# y = np.asarray(y,dtype='float')
#
# x = [i for i in range(0,len(ax))]
# ax.index = x
# ax = np.asarray(ax,dtype='float')
# rmse = np.sqrt(((y[0:96] - ax[0:96])**2))
#
# ax_plot.set_xlabel('Datetime')
# ax_plot.set_ylabel('Mobile Throughput')
# plt.legend()
# plt.xlim(0,96)
# plt.show()
#
# rmse_df = pd.DataFrame(rmse)
#
# print('standard deviation = {} ,mean = {}'.format(np.std(rmse_df),np.mean(rmse_df)))
# rmse_df[0].plot()   #plot root mean square error for first day
# plt.title('Root Mean Squared Error')
# plt.show()
#
# # #######
#
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return np.array(diff)


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


rolmean = pd.rolling_mean(subset['value'], window=12)
rolstd = pd.rolling_std(subset['value'], window=12)

# Plot rolling statistics:
orig = plt.plot(subset['value'], color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


subset_log = np.log(subset['value'])
moving_avg = pd.rolling_mean(subset_log,2)
plt.plot(subset_log,color = 'blue')
plt.plot(moving_avg, color='red')
plt.show()

diff = subset_log - moving_avg
print(diff.head(5))

dftest = adfuller(subset['value'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)



#diff.dropna(inplace=True)

diff[0] = 0
#test3 = test_stationarity(diff)



diff.index = pd.date_range(start = pd.to_datetime('2016-08-14 00:00:00'),periods=96, freq='15min')

subset_log.index = pd.date_range(start = pd.to_datetime('2016-08-14 00:00:00'),periods=95, freq='15min')

print (diff.index)
# mod = sm.tsa.ARIMA(diff,order = (0,1,6))
#
# results = mod.fit(disp=False)
#c
#
# results.plot_predict(start = pd.to_datetime('2016-08-14 00:15:00'),end =pd.to_datetime('2016-08-16 00:00:00'),alpha = 0.05)
#
# plt.show()
subset = subset.set_index('datetime')

# model = ARIMA(diff, order=(7, 1, 2))
# results_AR = model.fit(disp=-1)
# # plt.plot(diff, color = 'blue')
# # plt.plot(results_AR.fittedvalues, color='red')
# # plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-diff)**2))
#
# results_AR.plot_predict(start = pd.to_datetime('2016-08-14T00:15:00'),end =pd.to_datetime('2016-08-16T00:00:00'),alpha = 0.05)
# plt.show()


# lag_acf = acf(diff, nlags=20)
# lag_pacf = pacf(diff, nlags=20, method='ols')
#
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')
# plt.show()
#
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
# plt.show()

model1 = ARIMA(subset_log, order=(2, 1, 0))
results_AR = model1.fit(disp=-1)
plt.plot(diff,label = 'Differenced Series')
plt.plot(results_AR.fittedvalues, label = 'Predicted Values for Differenced Series', color='green')
plt.title('Forecasted Values for AR Model')#'RSS: %.4f'% sum((results_AR.fittedvalues-diff)**2))
plt.legend()
plt.show()

model2 = ARIMA(subset_log, order=(0, 1, 2))
results_MA = model2.fit(disp=-1)
plt.plot(diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('Forecasted Values for MA Model')#'RSS: %.4f'% sum((results_MA.fittedvalues-diff)**2))
plt.legend()
plt.show()

model = ARIMA(subset_log, order=(1, 0, 5))
results_ARIMA = model.fit(disp=-1)
# plt.plot(diff)
# plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('Forecasted Values for ARIMA Model')#'RSS: %.4f'% sum((results_ARIMA.fittedvalues-diff)**2))


results_ARIMA.plot_predict(start = pd.to_datetime('2016-08-14 15:00:00'),end = pd.to_datetime('2016-08-15 02:00:00'))
plt.show()
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(subset_log.ix[0], index=subset_log.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


predictions_ARIMA = np.exp(predictions_ARIMA_log)
#subset.plot(color = 'red')
predictions_ARIMA.plot(color = 'blue')

plt.show()

# plt.plot(subset,color = 'green')
# plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-subset)**2)/len(subset)))
# plt.show()

print('RMSE = ',np.sqrt(sum((predictions_ARIMA.values-subset.values)**2)/len(subset.values)))

rmse= np.sqrt(sum((np.asarray(predictions_ARIMA)-np.asarray(subset))**2)/len(subset))

rmse = pd.DataFrame(rmse)

rmse.plot()
plt.title('Root Mean Squared Error')
plt.show()
# predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
# predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# predictions_ARIMA_log = pd.Series(subset_log.index[0], index=subset_log.index)
# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
# predictions_ARIMA_log.head()
#
# predictions_ARIMA = np.exp(predictions_ARIMA_log)
# subset_log.index = pd.date_range(start = pd.to_datetime('2016-08-14 00:00:00'),periods=95, freq='15min')
# plt.plot(subset,color = 'blue')
# plt.plot(predictions_ARIMA,color = 'green')

# test3 = test_stationarity(subset)


# X = subset.values
# days_in_year = 1
# differenced = difference(X, days_in_year)
# # fit model
# model = sm.tsa.ARIMA(subset, order=(4,0,6))
# model_fit = model.fit(disp=0)
# # print summary of fit model
# print(model_fit.summary())
# #
# # forecast = model_fit.forecast(steps = 96*2,alpha = 0.05)[0]
# # forecast = inverse_difference(X, forecast, days_in_year)
# #
# # test = pd.DataFrame(forecast)
# # test.plot()
# # plt.show()
#
# model_fit.predict(start = pd.to_datetime('2016-12-31 00:00:00'),end = pd.to_datetime('2017-01-05 00:00:00')).plot()
#
# plt.show()
#
