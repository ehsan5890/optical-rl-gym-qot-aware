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
from multiprocessing import Process
import copy

# logging.getLogger("rmsaenv").setLevel(logging.INFO)

seed = 20
episodes = 1000
episode_length = 205

monitor_files = []
policies = []

logging_dir = "../examples/phy_frag_rmsa/jpn-results-new"
os.makedirs(logging_dir, exist_ok=True)

topology_name = 'jpn12'
# topology_name='nsfnet_chen'
with open(
        os.path.join("..", "examples", "topologies", f"{topology_name}_5-paths_6-modulations.h5"), "rb"
) as f:
    topology = pickle.load(f)

# mat_file = loadmat('../examples/inputs/Modulation_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
# mat_file1 = loadmat('../examples/inputs/GSNR_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
# mat_file2 = loadmat('../examples/inputs/All_connections_Profile_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
#
# modulation_jpn12 = mat_file['Modulation_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform']
# gsnr_jpn12 = mat_file1['GSNR_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform']
# all_connections_jpn12 = mat_file2['All_connections_Profile_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform']

mat_file3 = loadmat('../examples/inputs/Results_K3SP_FRP_SLC_CBG_USB14.mat')
mat_file4 = loadmat('../examples/inputs/Results_K3SP_FRP_SLC_CBG_SPN30.mat')
mat_file5 = loadmat('../examples/inputs/Results_K3SP_FRP_SLC_CBG_JPN12.mat')
jpn_data = mat_file5['Results_K3SP_FRP_SLC_CBG_JPN12']
modulation_jpn12 = jpn_data[0][0][1]
gsnr_jpn12 = jpn_data[0][0][2]
all_connections_jpn12 = jpn_data[0][0][0]

min_load = 800
max_load = 901
step_length = 100
steps = int((max_load - min_load) / step_length) + 1


def run_with_callback(callback, env_args, num_eps, log_dir):
    if callback is phy_aware_bmff_rmsa:
        env = gym.make("PhyRMSA-v0", **env_args)
        env = Monitor(env, log_dir + 'BM-SA-FF',
                      info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
                                     'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
                                     'total_path_length', 'num_moves',
                                     'num_defrag_cycle', 'avrage_gsnr', 'average_mod_level', 'average_path_index',
                                     'path_index', 'physical_paths'))
    elif callback is phy_aware_bmfa_rmsa:
        env = gym.make("PhyRMSA-v0", **env_args)
        env = Monitor(env, log_dir + 'BM-FA-Cut-modified',
                      info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
                                     'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
                                      'total_path_length',
                                     'num_moves',
                                     'num_defrag_cycle', 'avrage_gsnr', 'average_mod_level', 'average_path_index',
                                     'path_index', 'physical_paths'))

    elif callback is phy_aware_sapbm_rmsa:
        env = gym.make("PhyRMSA-v0", **env_args)
        env = Monitor(env, log_dir + 'BM-FA-SAPBM',
                      info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
                                     'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
                                      'total_path_length',
                                     'num_moves',
                                     'num_defrag_cycle', 'avrage_gsnr', 'average_mod_level', 'average_path_index',
                                     'path_index', 'physical_paths'))
    else:
        env = gym.make("PhyRMSA-v0", **env_args)
        env = Monitor(env, log_dir + 'BM-FA-RSS',
                      info_keywords=('episode_service_blocking_rate', 'service_blocking_rate',
                                     'episode_bit_rate_blocking_rate', 'number_cuts_total', 'rss_total_metric',
                                      'total_path_length',
                                     'num_moves', 'num_defrag_cycle', 'avrage_gsnr', 'average_mod_level',
                                     'average_path_index', 'path_index', 'physical_paths'))

    evaluate_heuristic(
        env, callback, n_eval_episodes=num_eps
    )


if __name__ == '__main__':
    processes = []
    for load_counter, load in enumerate(range(min_load, max_load, step_length)):
        env_args = dict(
            topology=topology,
            seed=10,
            allow_rejection=True,
            load=load,
            mean_service_holding_time=25,
            episode_length=episode_length,
            num_spectrum_resources=64,
            bit_rate_selection="discrete",
            modulation_level=modulation_jpn12,
            connections_detail=all_connections_jpn12,
            gsnr=gsnr_jpn12,
            number_spectrum_channels=80,
            number_spectrum_channels_s_band=108,
        )
        log_dir = f'{logging_dir}/logs_{load}_{episode_length}/'
        os.makedirs(log_dir, exist_ok=True)

        env_phy_df = gym.make("PhyRMSA-v0", **env_args)

        p = Process(target=run_with_callback, args=(phy_aware_bmff_rmsa, copy.deepcopy(env_args), episodes,log_dir))
        p.start()
        processes.append(p)


        p = Process(target=run_with_callback, args=(phy_aware_bmfa_rmsa, copy.deepcopy(env_args), episodes,log_dir))
        p.start()
        processes.append(p)

        p = Process(target=run_with_callback, args=(phy_aware_bmfa_rss_rmsa, copy.deepcopy(env_args), episodes,log_dir))
        p.start()
        processes.append(p)

        p = Process(target=run_with_callback, args=(phy_aware_sapbm_rmsa, copy.deepcopy(env_args), episodes,log_dir))
        p.start()
        processes.append(p)

        env_args_defrag = dict(
            topology=topology,
            seed=10,
            allow_rejection=True,
            load=load,
            mean_service_holding_time=25,
            episode_length=episode_length,
            num_spectrum_resources=64,
            bit_rate_selection="discrete",
            modulation_level=modulation_jpn12,
            connections_detail=all_connections_jpn12,
            gsnr=gsnr_jpn12,
            number_spectrum_channels=80,
            number_spectrum_channels_s_band=108,
            defrag_period=10,
            number_moves=10,

        )

        log_dir = f'{logging_dir}/logs_{load}_{episode_length}-defragmeentation-cut/'
        os.makedirs(log_dir, exist_ok=True)
        #
        #
        # p = Process(target=run_with_callback, args=(phy_aware_sapff_rmsa, copy.deepcopy(env_args_defrag), episodes,log_dir))
        # p.start()
        # processes.append(p)
        #
        #
        #
        #
        #
        # p = Process(target=run_with_callback, args=(phy_aware_bmfa_rmsa, copy.deepcopy(env_args_defrag), episodes,log_dir))
        # p.start()
        # processes.append(p)
        # #
        env_args_defrag_rss = dict(
            topology=topology,
            seed=10,
            allow_rejection=True,
            load=load,
            mean_service_holding_time=25,
            episode_length=episode_length,
            num_spectrum_resources=64,
            bit_rate_selection="discrete",
            modulation_level=modulation_jpn12,
            connections_detail=all_connections_jpn12,
            gsnr=gsnr_jpn12,
            number_spectrum_channels=80,
            number_spectrum_channels_s_band=108,
            defrag_period=10,
            number_moves=10,
            metric='rss'

        )
        #
        # log_dir = f'{logging_dir}/logs_{load}_{episode_length}-defragmeentation-rss/'
        # os.makedirs(log_dir, exist_ok=True)
        #
        # p = Process(target=run_with_callback,
        #             args=(phy_aware_bmfa_rss_rmsa, copy.deepcopy(env_args_defrag_rss), episodes, log_dir))
        # p.start()
        # processes.append(p)
        #
        # p = Process(target=run_with_callback,
        #             args=(phy_aware_bmff_rmsa, copy.deepcopy(env_args_defrag_rss), episodes, log_dir))
        # p.start()
        # processes.append(p)
    [p.join() for p in processes]  # wait for the completion of all processes