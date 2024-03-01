import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
# from IPython.core.display import clear_output
import numpy as np
import time

heuristic_log_dir = 'phy_frag_rmsa'

topology_name = 'Germany50'
topology_name = 'jpn12'
figures_floder = f'./{heuristic_log_dir}/figures'
os.makedirs(figures_floder, exist_ok=True)

min_load = 1300
max_load = 1301
step_length = 10
steps = int((max_load - min_load) / step_length) + 1
loads = np.zeros(steps)

metrics = [
    'episode_service_blocking_rate',
    'service_blocking_rate',
    'episode_bit_rate_blocking_rate',
    'number_cuts_total',
    'rss_total_metric',
    'C_BVTs',
    'L_BVTs',
    'S_BVTs',
    'total_path_length'
]

sap_ff_loads = {metric: [] for metric in metrics}
bm_fa_cut_loads = {metric: [] for metric in metrics}
bm_fa_rss_loads = {metric: [] for metric in metrics}
bm_sa_ff_loads = {metric: [] for metric in metrics}
traffic_type = 1

for load_counter, load_traffic in enumerate(range(min_load, max_load, step_length)):
    bm_fa_cut = pd.read_csv(
        f'./{heuristic_log_dir}/logs_{load_traffic}_200/BM-FA-Cut.monitor.csv',
        skiprows=1)
    bm_fa_rss = pd.read_csv(
        f'./{heuristic_log_dir}/logs_{load_traffic}_200/BM-FA-RSS.monitor.csv', skiprows=1)
    bm_sa_ff = pd.read_csv(
        f'./{heuristic_log_dir}/logs_{load_traffic}_200/BM-SA-FF.monitor.csv',
        skiprows=1)
    sap_ff = pd.read_csv(
        f'./{heuristic_log_dir}/logs_{load_traffic}_200/SAP-FF.monitor.csv',
        skiprows=1)
    loads[load_counter] = load_traffic
    for info in ['episode_service_blocking_rate','service_blocking_rate',
                'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
                'C_BVTs', 'L_BVTs', 'S_BVTs', 'total_path_length']:
        bm_fa_cut_loads[info].append(np.mean(bm_fa_cut[info]))
        #             exhuastic_highest_loads[info].append(np.mean(exhaustic_highest[info]))
        bm_fa_rss_loads[info].append(np.mean(bm_fa_rss[info]))
        bm_sa_ff_loads[info].append(np.mean(bm_sa_ff[info]))
        sap_ff_loads[info].append(np.mean(sap_ff[info]))

# print(loads)
# print(f"SBR for highest-first-fit is {highest_loads['service_blocking_per_hundred_arrivals']}")


# percentage = []
# percentage1 = []
# for i in range(len(no_df_loads['service_blocking_per_hundred_arrivals'])):
#     percentage.append((no_df_loads['service_blocking_per_hundred_arrivals'][i] -
#                        exhuastic_oldest_loads['service_blocking_per_hundred_arrivals'][i]) /
#                       no_df_loads['service_blocking_per_hundred_arrivals'][i])
# for i in range(len(no_df_loads['service_blocking_per_hundred_arrivals'])):
#     percentage1.append((oldest_loads['service_blocking_per_hundred_arrivals'][i] -
#                         exhuastic_oldest_loads['service_blocking_per_hundred_arrivals'][i]) /
#                        oldest_loads['service_blocking_per_hundred_arrivals'][i])

# print(f"the exhaustic vs. no df-{[round(item, 2) for item in percentage]}")
# print(f"the exhaustic vs. oldest-{[round(item, 2) for item in percentage1]}")

markersize = 7
for info in ['episode_service_blocking_rate','service_blocking_rate',
                'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
                'C_BVTs', 'L_BVTs', 'S_BVTs', 'total_path_length']:
    plt.figure()
    if info is 'service_blocking_rate' or info is 'service_blocking_rate':
        ax_bm_sa_ff = plt.semilogy(loads, bm_sa_ff_loads[info], label='No-SD', marker='X', markersize=markersize,
                             markeredgecolor='white')
        ax_bm_fa_cut = plt.semilogy(loads, bm_fa_cut_loads[info], label='OF-FF (10, 10)', marker='P', markersize=markersize,
                                 markeredgecolor='white')
        ax_bm_fa_rss = plt.semilogy(loads, bm_fa_rss_loads[info], label='HNoC (10, 10)', marker='o',
                                           markersize=markersize, markeredgecolor='white')
        ax_sap_ff = plt.semilogy(loads, sap_ff_loads[info], label='HRSS (10, 10)', marker='p', markersize=markersize,
                                  markeredgecolor='white')
    else:
        ax_bm_sa_ff = plt.plot(loads, bm_sa_ff_loads[info], label='No-SD', marker='X', markersize=markersize,
                         markeredgecolor='white')
        ax_bm_fa_cut = plt.plot(loads, bm_fa_cut_loads[info], label='OF-FF (10, 10)', marker='P', markersize=markersize,
                             markeredgecolor='white')
        ax_bm_fa_rss = plt.plot(loads, bm_fa_rss_loads[info], label='HNoC (10, 10) ', marker='o',
                                       markersize=markersize, markeredgecolor='white')
        ax_sap_ff = plt.plot(loads, sap_ff_loads[info], label='HRSS (10, 10)', marker='p', markersize=markersize,
                              markeredgecolor='white')

    print(f"the bm-sa-ff for {info} is {np.mean(bm_sa_ff_loads[info])} ")
    print(f"the bm_fa_cut for {info} is {np.mean(bm_fa_cut_loads[info])} ")
    print(f"the bm_fa_rss for {info} is {np.mean(bm_fa_rss_loads[info])} ")
    print(f"the sap_ff for {info} is {np.mean(sap_ff_loads[info])} ")

    plt.xlabel('Load [Erlang]')
    # if info == 'service_blocking_per_hundred_arrivals':
    #     plt.ylabel('Service Blocking Ratio (SBR)', fontsize=13)
    # elif info == 'episode_frag_metric':
    #     plt.ylabel('RSS metric', fontsize=13)
    # else:
    #     plt.ylabel(info)

    plt.legend(loc='upper right')
    plt.ylabel(info)
    plt.legend(fontsize=12)
    plt.rcParams.update({'font.size': 13})
    plt.savefig(f'{figures_floder}/{topology_name}-{info}.pdf')



