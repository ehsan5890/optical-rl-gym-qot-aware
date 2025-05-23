import logging
import os
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor
from scipy.io import loadmat
from optical_rl_gym.envs.phy_rmsa_env import (
    phy_aware_bmff_rmsa,
    phy_aware_bmfa_rmsa,
    phy_aware_bmfa_rss_rmsa,
    phy_aware_sapbm_rmsa

)
from optical_rl_gym.utils import evaluate_heuristic, random_policy


# logging.getLogger("rmsaenv").setLevel(logging.INFO)

seed = 20
episodes = 1
episode_length = 18000

monitor_files = []
policies = []

logging_dir = "../examples/phy_frag_rmsa/us-results"
os.makedirs(logging_dir, exist_ok=True)


# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
topology_name='us14'
# topology_name='nsfnet_chen'
with open(
    os.path.join("..", "examples", "topologies", f"{topology_name}_3-paths_6-modulations.h5"), "rb"
) as f:
    topology = pickle.load(f)

mat_file3 = loadmat('../examples/inputs/Results_K3SP_FRP_SLC_CBG_USB14.mat')

# US data info
us_data = mat_file3['Results_K3SP_FRP_SLC_CBG_USB14']
modulation_us = us_data[0][0][1]
gsnr_us = us_data[0][0][2]
all_connections_us = us_data[0][0][0]

min_load = 2200
max_load = 2201
step_length = 60
steps = int((max_load - min_load)/step_length) +1


for load_counter, load in enumerate(range(min_load,max_load,step_length)):
    env_args = dict(
        topology=topology,
        seed=10,
        allow_rejection=True,
        load=load,
        mean_service_holding_time=25,
        episode_length=episode_length,
        num_spectrum_resources=64,
        bit_rate_selection="discrete",
        modulation_level=modulation_us,
        connections_detail=all_connections_us,
        gsnr=gsnr_us,
        number_spectrum_channels=80,
        number_spectrum_channels_s_band=108,
        grooming=False,

    )



    print("STR".ljust(5), "REW".rjust(7), "STD".rjust(7))

    log_dir = f'{logging_dir}/logs_{load}_{episode_length}/'
    os.makedirs(log_dir, exist_ok=True)


    # env_phy_df = gym.make("PhyRMSA-v0", **env_args)
    # env_phy_df = Monitor(env_phy_df, log_dir + 'SAP-FF', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
    #                                                                                          'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
    #                                                                                          'C_BVTs', 'L_BVTs', 'S_BVTs', 'total_path_length'))
    #
    # mean_reward_sp, std_reward_sp = evaluate_heuristic(
    #     env_phy_df, phy_aware_sapff_rmsa, n_eval_episodes=episodes
    # )
    # print("SAP-FF:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
    # print(
    #     "\tBit rate blocking:",
    #     (env_phy_df.episode_bit_rate_requested - env_phy_df.episode_bit_rate_provisioned)
    #     / env_phy_df.episode_bit_rate_requested,
    # )
    # print(
    #     "\tRequest blocking:",
    #     (env_phy_df.episode_services_processed - env_phy_df.episode_services_accepted)
    #     / env_phy_df.episode_services_processed,
    # )
    #
    # env_phy_bmff_df = gym.make("PhyRMSA-v0", **env_args)
    # env_phy_bmff_df = Monitor(env_phy_bmff_df, log_dir + 'BM-SA-FF', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
    #                                                                                          'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
    #                                                                                           'total_path_length'))
    # mean_reward_sp, std_reward_sp = evaluate_heuristic(
    #     env_phy_bmff_df, phy_aware_bmff_rmsa, n_eval_episodes=episodes
    # )
    # print("BM-FF:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
    # print(
    #     "\tBit rate blocking:",
    #     (env_phy_bmff_df.episode_bit_rate_requested - env_phy_bmff_df.episode_bit_rate_provisioned)
    #     / env_phy_bmff_df.episode_bit_rate_requested,
    # )
    # print(
    #     "\tRequest blocking:",
    #     (env_phy_bmff_df.episode_services_processed - env_phy_bmff_df.episode_services_accepted)
    #     / env_phy_bmff_df.episode_services_processed,
    # )

    env_phy_bmfa_cut_df = gym.make("PhyRMSA-v0", **env_args)
    env_phy_bmfa_cut_df = Monitor(env_phy_bmfa_cut_df, log_dir + 'BM-FA-Cut', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
                                                                                             'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
                                                                                              'total_path_length', 'avrage_gsnr', 'average_mod_level'))
    mean_reward_sp, std_reward_sp = evaluate_heuristic(
        env_phy_bmfa_cut_df, phy_aware_bmfa_rmsa, n_eval_episodes=episodes
    )
    print("BM-FA:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
    print(
        "\tBit rate blocking:",
        (env_phy_bmfa_cut_df.episode_bit_rate_requested - env_phy_bmfa_cut_df.episode_bit_rate_provisioned)
        / env_phy_bmfa_cut_df.episode_bit_rate_requested,
    )
    print(
        "\tRequest blocking:",
        (env_phy_bmfa_cut_df.episode_services_processed - env_phy_bmfa_cut_df.episode_services_accepted)
        / env_phy_bmfa_cut_df.episode_services_processed,
    )

    # env_phy_bmfa_rss_df = gym.make("PhyRMSA-v0", **env_args)
    #
    # env_phy_bmfa_rss_df = Monitor(env_phy_bmfa_rss_df, log_dir + 'BM-FA-RSS', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
    #                                                                                          'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
    #                                                                                          'C_BVTs', 'L_BVTs', 'S_BVTs', 'total_path_length'))
    # mean_reward_sp, std_reward_sp = evaluate_heuristic(
    #     env_phy_bmfa_rss_df, phy_aware_bmfa_rss_rmsa, n_eval_episodes=episodes
    # )
    # print("BM-FA-Rss:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
    # print(
    #     "\tBit rate blocking:",
    #     (env_phy_bmfa_rss_df.episode_bit_rate_requested - env_phy_bmfa_rss_df.episode_bit_rate_provisioned)
    #     / env_phy_bmfa_rss_df.episode_bit_rate_requested,
    # )
    # print(
    #     "\tRequest blocking:",
    #     (env_phy_bmfa_rss_df.episode_services_processed - env_phy_bmfa_rss_df.episode_services_accepted)
    #     / env_phy_bmfa_rss_df.episode_services_processed,
    # )



    env_args_defrag = dict(
        topology=topology,
        seed=10,
        allow_rejection=True,
        load=load,
        mean_service_holding_time=25,
        episode_length=episode_length,
        num_spectrum_resources=64,
        bit_rate_selection="discrete",
        modulation_level=modulation_us,
        connections_detail=all_connections_us,
        gsnr=gsnr_us,
        number_spectrum_channels=80,
        number_spectrum_channels_s_band=108,
        defrag_period=10,
        number_moves=10,

    )

    # env_phy_bmfa_cut_df = gym.make("PhyRMSA-v0", **env_args_defrag)
    # env_phy_bmfa_cut_df = Monitor(env_phy_bmfa_cut_df, log_dir + 'BM-FA-Cut', info_keywords=('episode_service_blocking_rate','service_blocking_rate',
    #                                                                                          'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
    #                                                                                          'C_BVTs', 'L_BVTs', 'S_BVTs', 'total_path_length'))
    # mean_reward_sp, std_reward_sp = evaluate_heuristic(
    #     env_phy_bmfa_cut_df, phy_aware_bmfa_rmsa, n_eval_episodes=episodes
    # )
    # print("BM-FA-defrag:".ljust(8), f"{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}")
    # print(
    #     "\tBit rate blocking:",
    #     (env_phy_bmfa_cut_df.episode_bit_rate_requested - env_phy_bmfa_cut_df.episode_bit_rate_provisioned)
    #     / env_phy_bmfa_cut_df.episode_bit_rate_requested,
    # )
    # print(
    #     "\tRequest blocking:",
    #     (env_phy_bmfa_cut_df.episode_services_processed - env_phy_bmfa_cut_df.episode_services_accepted)
    #     / env_phy_bmfa_cut_df.episode_services_processed,
    # )
