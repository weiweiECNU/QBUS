import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dateparse(dates): return pd.datetime.strptime(dates, '%Y-%m')


data_time = pd.read_csv('AirPassengers.csv',
                        parse_dates=['Month'], index_col='Month', date_parser=dateparse)



# print(data_time)
ts = data_time['Passengers']

ts1 = ts['1949']



plt.plot(ts)
plt.title('Air Passenger 1949-1961number without Noise')
plt.xlabel('Time')
plt.ylabel('Passengers number')

np.random.seed(0)

rand_data = np.random.randn(len(ts))

ts_noise = ts + rand_data * 20

plt.figure()
plt.plot(ts_noise)
plt.title('Air Passenger 1949-1961number with Noise')
plt.xlabel('Time')
plt.ylabel('Passengers number')


data_shift = ts_noise.shift(12)
data_tshift = ts_noise.tshift(12)

fig, ax = plt.subplots(3, 1, sharex='col')
ax[0].plot(ts_noise)
ax[1].plot(data_shift)
ax[2].plot(data_tshift)

ax[0].legend(['noised data'], loc=2)
ax[1].legend(['shift(12)'], loc=2)
ax[2].legend(['tshift(12)'], loc=2)


local_date = pd.to_datetime('1950-01-01')
offset = pd.Timedelta(12, 'M')


ax[0].axvline(local_date, ymin=0.3, ymax=0.5, alpha=1, color='red')
ax[1].axvline(local_date + offset, alpha=0.3, color='red')
ax[2].axvline(local_date + offset, alpha=0.3, color='red')





