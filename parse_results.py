# coding: utf-8

# # Experimental Results
# This script parses experimental logs to obtain performance metrics.
# Please note that log_analyzer.py is used from the tools directory.
# Documentation of the usage for the LogAnalyzer class is provided on log_analyzer.py

# Import LogAnalyzer objects
from tools.log_analyzer import *
import numpy as np
# and other relevant stuff...
import matplotlib.pyplot as plt
from datetime import *
import tikzplotlib

# #### !IMPORTANT: Specify directory and log filenames here 
# Note that the provided names (below) are default names. They do not have to be changes unless you decided to rename files from multiple experiments.
log_dir_path = "./final_logs/Adapt2/sim/"
vehicle_log_file = "vehicle.log"
customer_log_file = "customer.log.1"
score_log_file = "score.log"
summary_log_file = "summary.log"


# Invoke LogAnalyzer Object
l = LogAnalyzer()


# #### Exploring dataframes of each of the logs
# Loading all the different logs as pandas dataframes
summary_df = l.load_summary_log(log_dir_path)
vehicle_df = l.load_vehicle_log(log_dir_path)
customer_df = l.load_customer_log(log_dir_path)
score_df = l.load_score_log(log_dir_path)

print(summary_df.describe())

print(vehicle_df.describe())

print(customer_df["waiting_time"].describe())

print(score_df.describe())


# #### Exploring the get_customer_status
df = l.get_customer_status(customer_df)
print(df.head())

df = l.get_customer_waiting_time(customer_df)
print(df.head())


# #### Generating plots of summary logs
DARM = "./final_logs/Adapt2/sim/"
DPRS = "./final_logs/NonAdapt2/sim/"
# Deep_Pool = "./baselines/Deep_Pool/sim/"
# ODA = "./baselines/ODA/sim/"
# Dummy_RS = "./baselines/Dummy_RS/sim/"
# Dummy_None = "./baselines/Dummy_None/sim/"

summary = l.load_summary_log("./logs/Adapt_replay/sim/")
print("Summary: ", summary)
stimes = pd.read_csv("./final_logs/switch_PC.txt")
print(len(stimes))
# print(type(stimes))

# stimes = (stimes / 3600).astype(int) * 3600
# summary = summary.groupby("t").mean().reset_index()
# times = list(stimes)
s_times = np.array(stimes, dtype=int)
# print(summary.t)
# print(s_times.tolist())

stimes = list(s_times)
# print(len(stimes))

req_change = dict()
# requests = summary[summary['t'].isin(stimes)]
for index, row in summary.iterrows():
	# print(row)
	if row.t in stimes:
		req_change[row.t] = row.n_requests
print(req_change)
print(len(req_change))
# print(len(requests))
# stimes = np.array(stimes[0:len(requests)], dtype=int)
stimes = np.array(list(req_change.keys()), dtype=int)
print(stimes)
# stimes = (stimes / 3600).astype(int) * 3600
# print(stimes)
s_times = [dt.datetime.fromtimestamp(t) for t in stimes]
# requests = requests.n_requests
# summary["day"] = [t.day for t in summary.t]
# print(s_times)
# print(requests)
years = md.YearLocator()  # every year
months = md.MonthLocator()  # every month
days = md.DayLocator()
hrs = md.HourLocator()
# xfmt = md.DateFormatter('%a: %m-%d')
xfmt = md.DateFormatter('| %a |')
# print(summary.t)


plt.ylabel("# of requests (Demand) at the detected change point")
plt.title("Change Points Detected in One Week Vs. Amount of Demand")
plt.xlabel("Simulation Time (hrs in days)")
plt.xticks(ha='left')
ax = plt.gca()
ax.set_xticks(s_times)
ax.xaxis.set_major_locator(days)
ax.xaxis.set_minor_locator(hrs)
ax.xaxis.set_major_formatter(xfmt)

# plt.legend(loc='lower right', framealpha = 0.7)
plt.plot(s_times, list(req_change.values()), 'o', color='blue')
# plt.xticks(rotation=50)
plt.savefig("./final_logs/switch.pdf")

sHrs = [dt.datetime.fromtimestamp(t) for t in stimes]
print("H: ", sHrs)
change_points = dict()
for h in sHrs:
	d = h.weekday()
	hr = h.hour
	# if d in change_points:
	# 	change_points[d].append(hr)
	# else:
	# 	change_points[d] = [hr]
	change_points[h] = int(hr)
	print(hr)

# print("Day, Hrs: ", change_points.keys(), change_points)
xfmt = md.DateFormatter('| %a |')
# print(summary.t)

# sHrs = np.array(list(change_points.keys()), dtype=int)
plt.ylabel("Detected change point (in hrs)")
plt.title("Change Points Detected in One Week of Evaluations")
plt.xlabel("Simulation Time (hrs in days)")
plt.xticks(ha='left')
ax = plt.gca()
ax.set_xticks(sHrs)
ax.xaxis.set_major_locator(days)
ax.xaxis.set_minor_locator(hrs)
ax.xaxis.set_major_formatter(xfmt)
print(change_points.values())
plt.ylim([-0.5, 24])

# i = 0
# txt = list(req_change.values())
# uni_y = list()
# for x,y in zip(sHrs,list(change_points.values())):
# 	label = str(txt[i]).format(y)
# 	i_x = x.weekday()
# 	# print(i_x)
# 	if (i_x,y) in uni_y:
# 		# print(label, i_x, y)
# 		i += 1
# 		continue
# 	else:
# 		uni_y.append((i_x,y))
# 		plt.annotate(label, # this is the text
# 				(x,y), # this is the point to label
# 				textcoords="offset points", # how to position the text
# 				xytext=(0,1), # distance from text to points (x,y)
# 				ha='center', color='black') # horizontal alignment can be left, right or center
# 	i += 1
plt.plot(sHrs, list(change_points.values()), '*', color='red')
# plt.xticks(rotation=50)
plt.savefig("./final_logs/switch_hrs.pdf")
# sys.exit()

summary_plots = l.plot_summary([DARM, DPRS], ["Number of Accepted Requests",
																"Avg. Travel Distance of Vehicles", "Occupancy "
																										"Rate of "
																							   "Vehicles"], plt)
summary_plots.savefig("./final_logs/Adapt2/Summary.pdf", bbox_inches = 'tight')
tikzplotlib.save("./final_logs/Adapt2/Summary.tex")
# tikzplotlib.clean_figure()
summary_plots.show()

# #### Generating plots of relevant experiment metrics
plt, df = l.plot_metrics([DARM], ["Profit", "Cruising Time", "Occupancy Rate","Waiting Time", "Travel "
																								"Distance"],plt)
plt.savefig("./final_logs/Adapt2/Metrics_Ada.pdf", bbox_inches = 'tight')
plt.show()

# #### We may also look at the metrics as a pandas dataframe
print(df.head())
