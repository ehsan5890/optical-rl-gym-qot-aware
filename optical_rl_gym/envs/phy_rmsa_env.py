import copy
import functools
import heapq
import logging
import math
from collections import defaultdict
from typing import Optional, Sequence, Tuple

import gym
import networkx as nx
import numpy as np
#from pygame.display import update

from optical_rl_gym.utils import Path, Service, Transponder

from .optical_network_env import OpticalNetworkEnv
from .rmsa_env import RMSAEnv


class PhyRMSAEnv(OpticalNetworkEnv):
    metadata = {
        "metrics": [
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
        ]
    }

    def __init__(
            self,
            topology: nx.Graph = None,
            episode_length: int = 1000,
            load: float = 10,
            mean_service_holding_time: float = 10800.0,
            num_spectrum_resources: int = 100,
            bit_rate_selection: str = "discrete",
            bit_rates: Sequence = [ 100, 200, 300, 400, 500, 600], #[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
            bit_rate_probabilities: Optional[np.array] = None,
            node_request_probabilities: Optional[np.array] = None,
            bit_rate_lower_bound: float = 25.0,
            bit_rate_higher_bound: float = 100.0,
            seed: Optional[int] = None,
            allow_rejection: bool = False,
            reset: bool = True,
            channel_width: float = 12.5,
            number_spectrum_channels: int = 80,
            number_spectrum_channels_s_band: int = 108,
            l_band: bool = True,
            s_band: bool = True,
            modulation_level: Optional[np.array] = None,
            connections_detail: Optional[np.array] = None,
            gsnr: Optional[np.array] = None,
            defrag_period=None,
            number_moves=None,
            metric='cut',
            grooming = True,
    ):
        super().__init__(
            topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            channel_width=channel_width,
            number_spectrum_channels=number_spectrum_channels,
            number_spectrum_channels_s_band=number_spectrum_channels_s_band,
            l_band=l_band,
            s_band=s_band,
        )

        # make sure that modulations are set in the topology
        assert "modulations" in self.topology.graph

        # asserting that the bit rate selection and parameters are correctly set
        assert bit_rate_selection in ["continuous", "discrete"]
        assert (bit_rate_selection == "continuous") or (
                bit_rate_selection == "discrete"
                and (
                        bit_rate_probabilities is None
                        or len(bit_rates) == len(bit_rate_probabilities)
                )
        )

        # specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.modulation_level = modulation_level
        self.connections_detail = connections_detail
        self.gsnr = gsnr
        self.defrag_period = defrag_period
        self.number_moves = number_moves
        self.metric = metric
        self.grooming = grooming

        self.number_cuts = 0
        self.rss_total_metric = 0
        self.total_path_length_episode = 0
        self.total_path_index_episode = 0
        self.total_gsnr_episode = 0
        self.total_modulation_level_episode = 0
        self.channels_accepted_episode = 0
        self.services_accepted_virtual = 0
        self.physical_services_accepted_episode = 0
        self.counted_moves = 0
        self.counted_moves_groom = 0
        self.counted_defrag_cycles = 0
        # setting up bit rate selection
        self.bit_rate_selection = bit_rate_selection
        if self.bit_rate_selection == "continuous":
            self.bit_rate_lower_bound = bit_rate_lower_bound
            self.bit_rate_higher_bound = bit_rate_higher_bound

            # creating a partial function for the bit rate continuous selection
            self.bit_rate_function = functools.partial(
                self.rng.randint, self.bit_rate_lower_bound, self.bit_rate_higher_bound
            )
        elif self.bit_rate_selection == "discrete":
            if bit_rate_probabilities is None:
                bit_rate_probabilities = [
                    1.0 / len(bit_rates) for x in range(len(bit_rates))
                ]
            self.bit_rate_probabilities = bit_rate_probabilities
            self.bit_rates = bit_rates

            # creating a partial function for the discrete bit rate options
            self.bit_rate_function = functools.partial(
                self.rng.choices, self.bit_rates, self.bit_rate_probabilities, k=1
            )

            # defining histograms which are only used for the discrete bit rate selection
            self.bit_rate_requested_histogram = defaultdict(int)
            self.bit_rate_provisioned_histogram = defaultdict(int)
            self.episode_bit_rate_requested_histogram = defaultdict(int)
            self.episode_bit_rate_provisioned_histogram = defaultdict(int)

            self.slots_requested_histogram = defaultdict(int)
            self.episode_slots_requested_histogram = defaultdict(int)
            self.slots_provisioned_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)

        self.spectrum_usage = np.zeros(
            (self.topology.number_of_edges(), self.num_spectrum_resources), dtype=int
        )
        self.grooming_layer = np.zeros(
            (self.topology.number_of_nodes(), self.topology.number_of_nodes(), self.k_paths), dtype=int
        )
        ## here 1: BVTs in C band, 0: BVTs in L band, 2: BVTs in S band
        self.bvts = np.zeros(
            (3, self.topology.number_of_nodes(), self.topology.number_of_nodes()), dtype=int
        )

        self.channel_state = np.empty((self.topology.number_of_nodes(), self.topology.number_of_nodes(), self.k_paths), dtype=object)
        for i in range(self.topology.number_of_nodes()):
            for j in range(self.topology.number_of_nodes()):
                for k in range(self.k_paths):
                    self.channel_state[i, j, k] = [] # the list consider tuples with (channel number, used by this connection , free, capacity, bool: True establish a connection, and false uses a connection)

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=int,
        )

        if self._s_band:
            self.spectrum_channels_allocation = np.full(
                (
                self.topology.number_of_edges(), 2 * self.num_spectrum_channels + self.number_spectrum_channels_s_band),
                fill_value=-1,
                dtype=int,
            )

            self.spectrum_channels_available = np.full(
                (
                self.topology.number_of_edges(), 2 * self.num_spectrum_channels + self.number_spectrum_channels_s_band),
                fill_value=-1,
                dtype=int,
            )

            self.topology.graph["available_channels"] = np.ones(
                (self.topology.number_of_edges(),
                 2 * self.num_spectrum_channels + self.number_spectrum_channels_s_band),
                dtype=int,
            )
        else:
            if self._l_band:
                self.spectrum_channels_allocation = np.full(
                    (self.topology.number_of_edges(),
                     2 * self.num_spectrum_channels),
                    fill_value=-1,
                    dtype=int,
                )

                self.spectrum_channels_available = np.full(
                    (self.topology.number_of_edges(),
                     2 * self.num_spectrum_channels),
                    fill_value=-1,
                    dtype=int,
                )
                self.topology.graph["available_channels"] = np.ones(
                    (self.topology.number_of_edges(),
                     2 * self.num_spectrum_channels),
                    dtype=int,
                )
            else:
                self.spectrum_channels_allocation = np.full(
                    (self.topology.number_of_edges(),
                     self.num_spectrum_channels),
                    fill_value=-1,
                    dtype=int,
                )

                self.spectrum_channels_available = np.full(
                    (self.topology.number_of_edges(),
                     self.num_spectrum_channels),
                    fill_value=-1,
                    dtype=int,
                )

                self.topology.graph["available_channels"] = np.ones(
                    (self.topology.number_of_edges(),
                     self.num_spectrum_channels),
                    dtype=int,
                )

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_output = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.actions_taken = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_taken = np.zeros(
            (self.k_paths + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.action_space = gym.spaces.MultiDiscrete(
            (
                self.k_paths + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            )
        )
        self.observation_space = gym.spaces.Dict(
            {
                "topology": gym.spaces.Discrete(10),
                "current_service": gym.spaces.Discrete(10),
            }
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger("rmsaenv")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of messages. "
                "Set it to INFO if DEBUG is not necessary."
            )
        self._new_service = False
        if reset:
            self.reset(only_episode_counters=False)

    def step(self, action):
        path, selected_channels = action[0], action[1]

        # starting the service as rejected
        self.current_service.accepted = False
        if (
                path != -2
        ):  # action is for assigning a path
            if path > 10:  ## service is allocated using virtual layer
                self.services_accepted_virtual +=1
                self._service_acceptance(True)
                self._provision_virtual_path(
                    self.k_shortest_paths[
                        self.current_service.source, self.current_service.destination
                    ][path-20],  # I wanted to know if this is virtual path or not
                    selected_channels
                )
            else:
                self.logger.debug(
                    "{} processing action {} path {} and initial slot {} for {} slots".format(
                        self.current_service.service_id, action, path, selected_channels, 2
                    )
                )
                if self.is_path_free_on_channels(
                        self.k_shortest_paths[
                            self.current_service.source, self.current_service.destination
                        ][path],
                        selected_channels
                ):
                    self._provision_path(
                        self.k_shortest_paths[
                            self.current_service.source, self.current_service.destination
                        ][path],
                        selected_channels
                    )
                    self._service_acceptance(False)

            self._add_release(self.current_service)
        self.topology.graph["services"].append(self.current_service)


        self.number_cuts = self._calculate_total_cuts()
        self.rss_total_metric = self.calculate_total_r_spatial()
        # self.total_path_length = 0
        # for service in self.topology.graph["running_services"]: ## It is not the best way.
        #     self.total_path_length += service.path.length
        reward = self.reward()
        info = {
            "service_blocking_rate": (self.services_processed - self.services_accepted)
                                     / self.services_processed,
            "episode_service_blocking_rate": (
                                                     self.episode_services_processed - self.episode_services_accepted
                                             )
                                             / self.episode_services_processed,
            "bit_rate_blocking_rate": (
                                              self.bit_rate_requested - self.bit_rate_provisioned
                                      )
                                      / self.bit_rate_requested,
            "episode_bit_rate_blocking_rate": (
                                                      self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
                                              )
                                              / self.episode_bit_rate_requested,
            "number_cuts_total": self.number_cuts,
            "rss_total_metric": self.rss_total_metric,
            # "C_BVTs": np.sum(self.bvts[1]) / (self.services_accepted + 1),
            # "L_BVTs": np.sum(self.bvts[0]) / (self.services_accepted + 1),
            # "S_BVTs": np.sum(self.bvts[2]) / (self.services_accepted + 1),
            "total_path_length": self.total_path_length_episode/( self.physical_services_accepted_episode + 1),
            "num_moves": self.counted_moves/2 + self.counted_moves_groom,
            "num_moves_groom": self.counted_moves_groom,
            "num_defrag_cycle": self.counted_defrag_cycles,
            "avrage_gsnr": self.total_gsnr_episode/(self.channels_accepted_episode + 1),
            "average_mod_level": self.total_modulation_level_episode/(self.channels_accepted_episode+1),
            "average_path_index": self.total_path_index_episode/( self.physical_services_accepted_episode + 1),
            "path_index": self.total_path_index_episode,
            "physical_paths":  self.physical_services_accepted_episode,
        }

        self._new_service = False
        self._next_service()
        if self.episode_services_processed == 30000:
            a = 1
        # Periodical defragmentation
        if self.defrag_period:
            if self.services_processed % self.defrag_period == 0:
                self.counted_moves_groom = self._groom_defragmentation()
                if self.counted_moves_groom <=self.number_moves:
                    defrag_candidates = []
                    ### there is no need to define a new channel class. go over running service, create the most good defragmentation options, then try to reallocate them.
                    for service in self.topology.graph["running_services"]:
                        links_indexes = []
                        for i in range(len(service.path.node_list) - 1):
                            links_indexes.append(
                                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]]["index"])
                        for channel in service.channels:
                            if channel[1] == channel[3]: ## we only reallocate full channels, the other ones propably is not appropriate to be reallocated
                                if self.metric == 'cut':
                                    cut_diff = self.calculate_r_cut(channel[0], links_indexes, True, service.path, True)
                                    if cut_diff > 0:
                                        defrag_candidates.append((cut_diff,
                                                                  self.current_time - service.arrival_time, channel[0],
                                                                  links_indexes, service, channel))
                                else:
                                    rss_diff = self.calculate_r_spatial(channel[0], links_indexes, True)
                                    if rss_diff > 0:
                                        defrag_candidates.append((rss_diff,
                                                                  self.current_time - service.arrival_time, channel[0],
                                                                  links_indexes, service, channel))
                    sorted_defrag_candidates = sorted(defrag_candidates, key=lambda x: (-x[0], -x[1]))
                    num_moves = 0
                    for i, candidate in enumerate(sorted_defrag_candidates):
                        reallocation_options = []
                        for channel_number in range(
                                0, self.topology.graph["num_channel_resources"]
                        ):
                            if self.is_channel_free(candidate[4].path, channel_number):
                                table_id = np.where(((self.connections_detail[:, 0] == int(candidate[4].source)) & (
                                        self.connections_detail[:, 1] == int(candidate[4].destination))) | (
                                                            (self.connections_detail[:, 0] == int(
                                                                candidate[4].destination)) & (
                                                                    self.connections_detail[:, 1] == int(
                                                                candidate[4].source))))[0][0]
                                for idp, serached_path in enumerate(
                                        self.k_shortest_paths[
                                            self.current_service.source, self.current_service.destination
                                        ]
                                ):
                                    if serached_path == candidate[4].path:
                                        break
                                if self.modulation_level[table_id][channel_number][idp] == self.modulation_level[table_id][candidate[2]][idp]:
                                    if self.metric == 'cut':
                                        fragmentation_metric = self.calculate_r_cut(channel_number, candidate[3], False, candidate[4].path, True )
                                    else:
                                        fragmentation_metric = self.calculate_r_spatial(channel_number, candidate[3])
                                    reallocation_options.append((fragmentation_metric, channel_number))
                        reallocation_options_sorted = sorted(reallocation_options, key=lambda x: (-x[0], x[1]))
                        if len(reallocation_options_sorted) > 0:  # the first options is chossed to be reallocated.
                            if -1 * reallocation_options_sorted[0][0] < candidate[
                                0]:  ## It makes sense to reallocate, since it gains in terms of fragmentation metric
                                self._move(candidate[4], reallocation_options_sorted[0][1], candidate[5])
                                num_moves += 1
                                self.counted_moves += 1
                        if num_moves + self.counted_moves_groom > self.number_moves:
                            break
                    if num_moves != 0:
                        self.counted_defrag_cycles += 1
        return (
            self.observation(),
            reward,
            self.episode_services_processed == self.episode_length,
            False,
            info,
        )

    def reset(self, only_episode_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.total_path_length_episode = 0
        self.total_path_index_episode = 0
        self.counted_defrag_cycles = 0
        self.counted_moves = 0
        self.counted_moves_groom = 0
        self.total_gsnr_episode = 0
        self.total_modulation_level_episode = 0
        self.channels_accepted_episode = 0
        self.physical_services_accepted_episode = 0
        self.episode_actions_output = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )
        self.episode_actions_taken = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )

        if self.bit_rate_selection == "discrete":
            self.episode_bit_rate_requested_histogram = defaultdict(int)
            self.episode_bit_rate_provisioned_histogram = defaultdict(int)
            self.episode_slots_requested_histogram = defaultdict(int)
            self.episode_slots_provisioned_histogram = defaultdict(int)

        if only_episode_counters:
            if self._new_service:
                # initializing episode counters
                # note that when the environment is reset, the current service remains the same and should be accounted for
                self.episode_services_processed += 1
                self.episode_bit_rate_requested += self.current_service.bit_rate
                if self.bit_rate_selection == "discrete":
                    self.episode_bit_rate_requested_histogram[
                        self.current_service.bit_rate
                    ] += 1

            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges(), self.num_spectrum_resources), dtype=int
        )

        if self._s_band:
            self.topology.graph["available_channels"] = np.ones(
                (self.topology.number_of_edges(),
                 2 * self.num_spectrum_channels + self.number_spectrum_channels_s_band),
                dtype=int,
            )
            self.spectrum_channels_allocation = np.full(
                (
                self.topology.number_of_edges(), 2 * self.num_spectrum_channels + self.number_spectrum_channels_s_band),
                fill_value=-1,
                dtype=int,
            )
        else:
            if self._l_band:
                self.topology.graph["available_channels"] = np.ones(
                    (self.topology.number_of_edges(),
                     2 * self.num_spectrum_channels),
                    dtype=int,
                )
                self.spectrum_channels_allocation = np.full(
                    (self.topology.number_of_edges(),
                     2 * self.num_spectrum_channels),
                    fill_value=-1,
                    dtype=int,
                )
            else:
                self.topology.graph["available_channels"] = np.ones(
                    (self.topology.number_of_edges(),
                     self.num_spectrum_channels),
                    dtype=int,
                )
                self.spectrum_channels_allocation = np.full(
                    (self.topology.number_of_edges(),
                     self.num_spectrum_channels),
                    fill_value=-1,
                    dtype=int,
                )

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=int,
        )

        # if self.bit_rate_selection == "discrete":
        #     self.bit_rate_requested_histogram = defaultdict(int)
        #     self.bit_rate_provisioned_histogram = defaultdict(int)

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0
        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["external_fragmentation"] = 0.0
            self.topology[lnk[0]][lnk[1]]["compactness"] = 0.0

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode="human"):
        return

    def _provision_path(self, path: Path, channels: list):
        # usage

        if not self.is_path_free_on_channels(path, channels):
            raise ValueError(
                "Path {} has not enough capacity on slots {}-{}".format(
                    path.node_list, path, channels
                )
            )

        self.logger.debug(
            "{} assigning path {} on initial slot {} for {} slots".format(
                self.current_service.service_id,
                path.node_list,
                channels,
                2,
            )
        )
        table_id = np.where(((self.connections_detail[:, 0] == int(self.current_service.source)) & (
                self.connections_detail[:, 1] == int(self.current_service.destination))) | (
                         (self.connections_detail[:, 0] == int(self.current_service.destination)) & (
                         self.connections_detail[:, 1] == int(self.current_service.source))))[0][0]

        for idp, serached_path in enumerate(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ]
        ):
            if serached_path == path:
                break
        path.idp = idp

        for i in range(len(path.node_list) - 1):
            for channel in channels:
                self.topology.graph["available_channels"][
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["index"],
                    channel[0],
                ] = 0
                self.spectrum_channels_allocation[
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["index"],
                    channel[0],
                ] = self.current_service.service_id

            self.topology[path.node_list[i]][path.node_list[i + 1]]["services"].append(
                self.current_service
            )
            self.topology[path.node_list[i]][path.node_list[i + 1]][
                "running_services"
            ].append(self.current_service)

        for i, channel in enumerate(channels):
            mod_level = channel[3]
            gsnr = self.gsnr[table_id][channel[0]][idp]
            self.total_gsnr_episode += gsnr
            self.total_modulation_level_episode += mod_level
            self.channels_accepted_episode += 1
            if channel[2] != 0:
                self.channel_state[self.current_service.source_id, self.current_service.destination_id, idp].append((channel[0], channel[1], channel[2], channel[3])
                    )
            if channel[0] <= self.num_spectrum_channels:
                self.bvts[1][self.current_service.source_id][self.current_service.destination_id] += 1
            elif self.num_spectrum_channels < channel[0] <= 2 * self.num_spectrum_channels:
                self.bvts[0][self.current_service.source_id][self.current_service.destination_id] += 1
            elif 2 * self.num_spectrum_channels < channel[0] < 2 * self.num_spectrum_channels + self.number_spectrum_channels_s_band:
                self.bvts[2][self.current_service.source_id][self.current_service.destination_id] += 1


        my_list = self.channel_state[self.current_service.source_id, self.current_service.destination_id, idp]
        first_elements = set()

        # Check for duplicate first elements
        for tup in my_list:
            if tup[0] in first_elements:
                print(f"Found two tuples with the same first element: {tup[0]}")
            first_elements.add(tup[0])

        self.topology.graph["running_services"].append(self.current_service)
        self.current_service.path = path
        self.current_service.channels = channels
        # self._update_network_stats()

    def _provision_virtual_path(self, path: Path, channels: list):
        for idp, serached_path in enumerate(
                self.k_shortest_paths[
                    self.current_service.source, self.current_service.destination
                ]
        ):
            if serached_path == path:
                break
        path.idp = idp

        for i, channel in enumerate(channels):
            target_channel1 = [t for t in self.channel_state[self.current_service.source_id, self.current_service.destination_id, idp] if t[0] == channel[0]]
            if len(target_channel1) >1:
                a=2
            target_channel = target_channel1[0]
            if target_channel[2] >= channel[1]: # It means there is enough free resource on the channel.
                self.channel_state[self.current_service.source_id, self.current_service.destination_id, idp].remove(target_channel)
                updated_channel = (target_channel[0], target_channel[1] + channel[1], target_channel[2] - channel[1], target_channel[3])
                self.channel_state[self.current_service.source_id, self.current_service.destination_id, idp].append(updated_channel)
            else:
                raise ValueError(f"Expected first element to be 1, but got error ")
        my_list = self.channel_state[self.current_service.source_id, self.current_service.destination_id, idp]
        first_elements = set()

        # Check for duplicate first elements
        for tup in my_list:
            if tup[0] in first_elements:
                print(f"Found two tuples with the same first element: {tup[0]}")
            first_elements.add(tup[0])


        self.topology.graph["running_services"].append(self.current_service)
        self.current_service.path = path
        self.current_service.channels = channels
        # self._update_network_stats()

    ### we remove and add the running services for future use!
    def _move(self, service: Service, new_channel_number, old_channel):

        for i in range(len(service.path.node_list) - 1):
            self.topology.graph["available_channels"][
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]]["index"],
                new_channel_number,
            ] = 0
            self.spectrum_channels_allocation[
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]]["index"],
                new_channel_number,
            ] = service.service_id

            self.topology.graph["available_channels"][
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                    "index"
                ],
                old_channel[0],
            ] = 1
            self.spectrum_channels_allocation[
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                    "index"
                ],
                old_channel[0],
            ] = -1
            self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                "running_services"
            ].remove(service)
        self.topology.graph["running_services"].remove(service)
        service.channels.remove(old_channel)
        new_channel = (new_channel_number, old_channel[1], old_channel[2], old_channel[3], old_channel[4])
        service.channels.append(new_channel)
        for i in range(len(service.path.node_list) - 1):
            self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                "running_services"
            ].append(service)
        self.topology.graph["running_services"].append(service)




    # we use this function to groom services before defragmentation
    def _groom_defragmentation(self):
        grooming_moves = 0
        for service in self.topology.graph["running_services"]:
            for channel in service.channels:
                if channel[1] != channel[3]:
                    # result = next((tup for tup in self.channel_state[service.source_id,
                    #                                                  service.destination_id,
                    #                                                  service.path.idp] if tup[0] == channel[0]), None)
                    corresponding_channel = [t for t in self.channel_state[
                        service.source_id, service.destination_id, service.path.idp] if t[0] == channel[0]][0]
                    if corresponding_channel[1] == channel[1]:  ###if used ones are the same, so we are eligible to do grooming,
                # because it does not affect any other services, and it makes sense to reallocate this connection. this is very important point.
                        for target_channel in self.channel_state[service.source_id, service.destination_id, service.path.idp]:
                            if target_channel != corresponding_channel:
                                if target_channel[2] >= channel[1]: # I can do grooming, two channels are removed and one is added.
                                    self.channel_state[
                                        service.source_id, service.destination_id, service.path.idp].remove(
                                        target_channel)
                                    self.channel_state[
                                        service.source_id, service.destination_id, service.path.idp].remove(
                                        corresponding_channel)
                                    updated_channel = (target_channel[0], target_channel[1] + channel[1], target_channel[2] - channel[1],target_channel[3])
                                    self.channel_state[
                                        service.source_id, service.destination_id, service.path.idp].append(updated_channel)
                                    new_channel = (target_channel[0], channel[1], updated_channel[2], target_channel[3], True)
                                    self._move_virtual(service, new_channel, channel)
                                    grooming_moves += 1
                                    break
                if grooming_moves ==self.number_moves:
                    return grooming_moves
        return grooming_moves

    def _move_virtual(self, service: Service, new_channel, old_channel):

        # for i in range(len(service.path.node_list) - 1):
        #     self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
        #         "running_services"
        #     ].remove(service)
        for i in range(len(service.path.node_list) - 1):
            self.topology.graph["available_channels"][
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                    "index"
                ],
                old_channel[0],
            ] = 1
            self.spectrum_channels_allocation[
                self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                    "index"
                ],
                old_channel[0],
            ] = -1
            # self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
            #     "running_services"
            # ].remove(service)
        self.topology.graph["running_services"].remove(service)
        service.channels.remove(old_channel)
        service.channels.append(new_channel) ### the channel of the service got changed
        # for i in range(len(service.path.node_list) - 1):
        #     self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
        #         "running_services"
        #     ].append(service)
        self.topology.graph["running_services"].append(service)


    def _service_acceptance(self, virtual: bool):
        self.current_service.virtual_layer = virtual
        self.current_service.accepted = True
        self.services_accepted += 1
        self.episode_services_accepted += 1

        if not virtual:
            self.total_path_length_episode += self.current_service.path.length
            self.total_path_index_episode += self.current_service.path.idp+1
            self.physical_services_accepted_episode += 1
        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned += self.current_service.bit_rate


    def _release_path(self, service: Service):
        for i in range(len(service.path.node_list) - 1):
            for channel in service.channels:
                if channel[1] == channel[3]:
                    self.topology.graph["available_channels"][
                        self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                            "index"
                        ],
                        channel[0],
                    ] = 1
                    self.spectrum_channels_allocation[
                        self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                            "index"
                        ],
                        channel[0],
                    ] = -1
            # self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
            #             "running_services"
            #         ].remove(service)        ####3 this part prbably should be added

        for channel in service.channels:
            if channel[1] != channel[3]:
                # result = next((tup for tup in self.channel_state[service.source_id,
                #                                                  service.destination_id,
                #                                                  service.path.idp] if tup[0] == channel[0]), None)
                result_temporarily =[t for t in self.channel_state[service.source_id, service.destination_id, service.path.idp] if t[0] == channel[0]]
                if len(result_temporarily) > 1:
                    a = 1
                result = result_temporarily[0]
                # if result [0] ==8:
                #     a = 2
                # if result not in self.channel_state[service.source_id,
                #                                                  service.destination_id,
                #                                                  service.path.idp]:
                #     a=1
                self.channel_state[
                    service.source_id, service.destination_id, service.path.idp].remove(result)
                if result[1] == channel[1]: ###if used ones are the same, so we are eligible to remove the channel
                    for i in range(len(service.path.node_list) - 1):
                        self.topology.graph["available_channels"][
                            self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                                "index"
                            ],
                            channel[0],
                        ] = 1
                        self.spectrum_channels_allocation[
                            self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                                "index"
                            ],
                            channel[0],
                        ] = -1
                        # self.topology[service.path.node_list[i]][service.path.node_list[i + 1]][
                        #     "running_services"
                        # ].remove(service)


                else:
                    self.channel_state[
                        service.source_id, service.destination_id, service.path.idp].append(
                        (result[0], result[1] - channel[1], result[2] + channel[1], result[3]))
                    for a in self.channel_state[
                        service.source_id, service.destination_id, service.path.idp]:
                        if a[1] == 0:
                            b =1


        my_list = self.channel_state[
                        service.source_id, service.destination_id, service.path.idp]
        first_elements = set()

        # Check for duplicate first elements
        for tup in my_list:
            if tup[0] in first_elements:
                print(f"Found two tuples with the same first element: {tup[0]}")
            first_elements.add(tup[0])

            # self._update_link_stats(
            #     service.path.node_list[i], service.path.node_list[i + 1]
            # )

        self.topology.graph["running_services"].remove(service)

    def _update_network_stats(self):
        last_update = self.topology.graph["last_update"]
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph["throughput"]
            last_compactness = self.topology.graph["compactness"]

            cur_throughput = 0.0

            for service in self.topology.graph["running_services"]:
                cur_throughput += service.bit_rate

            throughput = (
                                 (last_throughput * last_update) + (cur_throughput * time_diff)
                         ) / self.current_time
            self.topology.graph["throughput"] = throughput

            compactness = (
                                  (last_compactness * last_update)
                                  + (self._get_network_compactness() * time_diff)
                          ) / self.current_time
            self.topology.graph["compactness"] = compactness

        self.topology.graph["last_update"] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        last_update = self.topology[node1][node2]["last_update"]
        time_diff = self.current_time - self.topology[node1][node2]["last_update"]
        if self.current_time > 0:
            last_util = self.topology[node1][node2]["utilization"]
            cur_util = (
                               self.num_spectrum_resources
                               - np.sum(
                           self.topology.graph["available_slots"][
                           self.topology[node1][node2]["index"], :
                           ]
                       )
                       ) / self.num_spectrum_resources
            utilization = (
                                  (last_util * last_update) + (cur_util * time_diff)
                          ) / self.current_time
            self.topology[node1][node2]["utilization"] = utilization

            slot_allocation = self.topology.graph["available_slots"][
                              self.topology[node1][node2]["index"], :
                              ]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2][
                "external_fragmentation"
            ]
            last_compactness = self.topology[node1][node2]["compactness"]

            cur_external_fragmentation = 0.0
            cur_link_compactness = 0.0
            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RMSAEnv.rle(slot_allocation)

                # computing external fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1.0 - (
                        float(max_empty) / float(np.sum(slot_allocation))
                )

                # computing link spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6421472
                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = (
                            initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                    )

                    # evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                        slot_allocation[lambda_min:lambda_max]
                    )
                    unused_spectrum_slots = np.sum(1 - internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = (
                                                       (lambda_max - lambda_min) / np.sum(1 - slot_allocation)
                                               ) * (1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.0
                else:
                    cur_link_compactness = 1.0

            external_fragmentation = (
                                             (last_external_fragmentation * last_update)
                                             + (cur_external_fragmentation * time_diff)
                                     ) / self.current_time
            self.topology[node1][node2][
                "external_fragmentation"
            ] = external_fragmentation

            link_compactness = (
                                       (last_compactness * last_update) + (cur_link_compactness * time_diff)
                               ) / self.current_time
            self.topology[node1][node2]["compactness"] = link_compactness

        self.topology[node1][node2]["last_update"] = self.current_time

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(
            1 / self.mean_service_inter_arrival_time
        )
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # generate the bit rate according to the selection adopted
        bit_rate = (
            self.bit_rate_function()
            if self.bit_rate_selection == "continuous"
            else self.bit_rate_function()[0]
        )

        self.current_service = Service(
            self.episode_services_processed,
            src,
            src_id,
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate,
        )
        self._new_service = True

        self.services_processed += 1
        self.episode_services_processed += 1

        # registering statistics about the bit rate requested
        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate
        if self.bit_rate_selection == "discrete":
            self.bit_rate_requested_histogram[bit_rate] += 1
            self.episode_bit_rate_requested_histogram[bit_rate] += 1

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                #if not service_to_release.virtual_layer:  # in the previous version, the services were not released, but here they should be releases
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

    def is_path_free_on_channels(self, path: Path, selected_channels: list) -> bool:
        free_flag = True
        for channel_number in selected_channels:
            for i in range(len(path.node_list) - 1):
                if self.topology.graph["available_channels"][
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["index"],
                    channel_number[0]] == 0:
                    free_flag = False
        return free_flag

    def is_channel_free(self, path: Path, channel_number: int) -> bool:
        for i in range(len(path.node_list) - 1):
            if self.topology.graph["available_channels"][
                self.topology[path.node_list[i]][path.node_list[i + 1]]["index"],
                channel_number] == 0:
                return False
        return True

    def rle(inarray):
        """run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)"""
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][path]
        )

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][path]
        )

        # getting the blocks
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[: self.j]

        return initial_indices[final_indices], lengths[final_indices]

    def calculate_r_spatial(self, channel_number, link_indexes, defrag_flag: bool = False):
        r_spatial = 0
        r_spatial_after = 0
        initial_indices, values, lengths = \
            RMSAEnv.rle(self.topology.graph['available_channels'][:, channel_number])
        unused_blocks = [i for i, x in enumerate(values) if x == 1]
        r_spatial += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) / (
                    np.sum(lengths[unused_blocks]) + 1)  # added to one to avoid infinity?

        temporary_channels = copy.deepcopy(self.topology.graph['available_channels'][:, channel_number])
        if defrag_flag:
            for i in link_indexes:
                temporary_channels[i] = 1
        else:
            for i in link_indexes:
                temporary_channels[i] = 0

        initial_indices_after, values_after, lengths_after = \
            RMSAEnv.rle(temporary_channels)

        unused_blocks_after = [i for i, x in enumerate(values_after) if x == 1]
        r_spatial_after += np.sqrt(np.sum(lengths_after[unused_blocks_after] ** 2)) / (
                np.sum(lengths_after[unused_blocks_after]) + 1)  # added to one to avoid infinity?
        return r_spatial_after - r_spatial

    def calculate_total_r_spatial(self):
        r_spatial = 0
        for channel_number in range(
                self.num_spectrum_channels + self.num_spectrum_channels + self.number_spectrum_channels_s_band):  # c + L + S band
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_channels'][:, channel_number])
            unused_blocks = [i for i, x in enumerate(values) if x == 1]
            r_spatial += np.sqrt(np.sum(lengths[unused_blocks] ** 2)) / (
                        np.sum(lengths[unused_blocks]) + 1)  # added to one to avoid infinity?

        return r_spatial / (
                    self.num_spectrum_channels + self.num_spectrum_channels + self.number_spectrum_channels_s_band)
    ## we have modified it in a way that it only considers the neighbor links for calculating the number of cuts
    def calculate_r_cut(self, channel_number, link_indexes, defrag_flag: bool = False, path: Path = None, modified = False ):
        # defrag_flag is used to check when this function has been called, in the defragmentation process
        # or in the fragmentation aware process.
        if not modified:

            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_channels'][:, channel_number])
            temporary_channels = copy.deepcopy(self.topology.graph['available_channels'][:, channel_number])
            if defrag_flag:
                for i in link_indexes:
                    temporary_channels[i] = 1
            else:
                for i in link_indexes:
                    temporary_channels[i] = 0
            initial_indices_after, values_after, lengths_after = \
                RMSAEnv.rle(temporary_channels)
            return np.sum(values) - np.sum(values_after)
        else:

            temporary_channels1 = copy.deepcopy(self.topology.graph['available_channels'][:, channel_number])

            number_cut_before = 0
            number_cut_after = 0
            for i in range(len(path.node_list)) :
                for node_key in self.topology[path.node_list[i]]:
                    if node_key not in path.node_list:
                    # indexes_set.add(self.topology[path.node_list[i]][node_key]["index"])
                        if i == len(path.node_list) -1:
                            number_cut_before += abs(
                                temporary_channels1[self.topology[path.node_list[i-1]][path.node_list[i]]["index"]]
                                - temporary_channels1[self.topology[path.node_list[i]][node_key]["index"]])
                        elif i == 0:
                            number_cut_before += abs(temporary_channels1[self.topology[path.node_list[i]][path.node_list[i+1]]["index"]]
                                                     -temporary_channels1[self.topology[path.node_list[i]][node_key]["index"]])
                        else:
                            number_cut_before += abs(
                                temporary_channels1[self.topology[path.node_list[i]][path.node_list[i + 1]]["index"]]
                                - temporary_channels1[self.topology[path.node_list[i]][node_key]["index"]]) + abs(
                                temporary_channels1[self.topology[path.node_list[i-1]][path.node_list[i]]["index"]]
                                - temporary_channels1[self.topology[path.node_list[i]][node_key]["index"]]
                            )


            temporary_channels2 = copy.deepcopy(self.topology.graph['available_channels'][:, channel_number])
            if defrag_flag:
                for i in link_indexes:
                    temporary_channels2[i] = 1
            else:
                for i in link_indexes:
                    temporary_channels2[i] = 0

            for i in range(len(path.node_list)) :
                for node_key in self.topology[path.node_list[i]]:
                    if node_key not in path.node_list:
                    # indexes_set.add(self.topology[path.node_list[i]][node_key]["index"])
                        if i == len(path.node_list) -1:
                            number_cut_after += abs(
                                temporary_channels2[self.topology[path.node_list[i-1]][path.node_list[i]]["index"]]
                                - temporary_channels2[self.topology[path.node_list[i]][node_key]["index"]])
                        elif i == 0:
                            number_cut_after += abs(temporary_channels2[self.topology[path.node_list[i]][path.node_list[i+1]]["index"]]
                                                     -temporary_channels2[self.topology[path.node_list[i]][node_key]["index"]])
                        else:
                            number_cut_after += abs(
                                temporary_channels2[self.topology[path.node_list[i]][path.node_list[i + 1]]["index"]]
                                - temporary_channels2[self.topology[path.node_list[i]][node_key]["index"]]) + abs(
                                temporary_channels2[self.topology[path.node_list[i-1]][path.node_list[i]]["index"]]
                                - temporary_channels2[self.topology[path.node_list[i]][node_key]["index"]]
                            )

            return number_cut_before - number_cut_after

    def _calculate_total_cuts(self):
        number_cut = 0
        for channel_number in range(
                self.num_spectrum_channels + self.num_spectrum_channels + self.number_spectrum_channels_s_band):  # c + L + S band
            initial_indices, values, lengths = \
                RMSAEnv.rle(self.topology.graph['available_channels'][:, channel_number])
            number_cut += np.sum(values)
        return number_cut / (
                    self.num_spectrum_channels + self.num_spectrum_channels + self.number_spectrum_channels_s_band)

    def _get_network_compactness(self):
        # implementing network spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6476152

        sum_slots_paths = 0  # this accounts for the sum of all Bi * Hi

        for service in self.topology.graph["running_services"]:
            sum_slots_paths += service.number_slots * service.path.hops

        # this accounts for the sum of used blocks, i.e.,
        # \sum_{j=1}^{M} (\lambda_{max}^j - \lambda_{min}^j)
        sum_occupied = 0

        # this accounts for the number of unused blocks \sum_{j=1}^{M} K_j
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            # getting the blocks
            initial_indices, values, lengths = RMSAEnv.rle(
                self.topology.graph["available_slots"][
                self.topology[n1][n2]["index"], :
                ]
            )
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += (
                        lambda_max - lambda_min
                )  # we do not put the "+1" because we use zero-indexed arrays

                # evaluate again only the "used part" of the spectrum
                internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                    self.topology.graph["available_slots"][
                    self.topology[n1][n2]["index"], lambda_min:lambda_max
                    ]
                )
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0:
            cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (
                    self.topology.number_of_edges() / sum_unused_spectrum_blocks
            )
        else:
            cur_spectrum_compactness = 1.0

        return cur_spectrum_compactness



def phy_aware_sapbm_rmsa(env: PhyRMSAEnv) -> Tuple[int, list]:
    # first to check if we have enough resource in virtual layer
    result = use_existing_channels(env)
    if result[0] != -3:
        return (result[0] + 20, result[1])
    else:
        unassigned_bitrate = env.current_service.bit_rate
        selected_channels = []
        # free_channels = []
        table_id = np.where(((env.connections_detail[:, 0] == int(env.current_service.source)) & (
                    env.connections_detail[:, 1] == int(env.current_service.destination))) | (
                                    (env.connections_detail[:, 0] == int(env.current_service.destination)) & (
                                        env.connections_detail[:, 1] == int(env.current_service.source))))[0][0]
        free_channels = [[] for _ in range(env.k_paths)]

        ## this could be the Shortest available path first fit solution.
        for idp, path in enumerate(
                env.k_shortest_paths[
                    env.current_service.source, env.current_service.destination
                ]
        ):
            for channel_number in range(
                    0, env.topology.graph["num_channel_resources"]
            ):
                if env.is_channel_free(path, channel_number):
                    mod_level = env.modulation_level[table_id][channel_number][idp]
                    free_channels[idp].append((mod_level, channel_number, idp))

        sorted_free_channels = [sorted(row, key=lambda x: (-x[0], x[1])) for row in free_channels]

        while True:
            max_modulation_level = float('-inf')
            max_modulation_row_index = None
            unassigned_bitrate = env.current_service.bit_rate
            selected_channels = []
            for i in range(len(sorted_free_channels) - 1, -1, -1):
                # Check if the modulation level of the first tuple in the row is greater than or equal to the current maximum
                if sorted_free_channels[i] ==[]:
                    sorted_free_channels.pop(i)

            if len(sorted_free_channels) > 0:
                max_modulation_row_index = 0
                try:
                    max_modulation_level = sorted_free_channels[0][0][0]
                except:
                    aaa=1
            if max_modulation_row_index is None:
                return (-2, [])
            else:
                for channel in sorted_free_channels[max_modulation_row_index]:
                    unassigned_bitrate -= int(channel[0]) * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((channel[1], channel[0] + unassigned_bitrate/100, unassigned_bitrate/-100, channel[0], False ))
                        return (channel[2], selected_channels)
                    else:
                        selected_channels.append((channel[1], channel[0], 0, channel[0], False))
                if len(sorted_free_channels) > max_modulation_row_index:
                    sorted_free_channels.pop(max_modulation_row_index)
                elif len(sorted_free_channels) == 0:
                    return (-2, [])
        return (-2, [])  # -2 means service is blocked


def phy_aware_bmff_rmsa(env: PhyRMSAEnv) -> Tuple[int, list]:
    # first to check if we have enough resource in virtual layer
    # TODO: this is not the best way to handle the grroming layer. For example, if grooming layer has 300 Gbps, and service is 500 Gbps, then 300 can be used!
    # first to check if we have enough resource in virtual layer
    result = use_existing_channels(env)
    if result[0] != -3:
        return (result[0] + 20, result[1])
    else:
        table_id = np.where(((env.connections_detail[:, 0] == int(env.current_service.source)) & (
                    env.connections_detail[:, 1] == int(env.current_service.destination))) | (
                                    (env.connections_detail[:, 0] == int(env.current_service.destination)) & (
                                        env.connections_detail[:, 1] == int(env.current_service.source))))[0][0]
        free_channels = [[] for _ in range(env.k_paths)]

        ## this could be the Shortest available path first fit solution.
        for idp, path in enumerate(
                env.k_shortest_paths[
                    env.current_service.source, env.current_service.destination
                ]
        ):
            for channel_number in range(
                    0, env.topology.graph["num_channel_resources"]
            ):
                if env.is_channel_free(path, channel_number):
                    mod_level = env.modulation_level[table_id][channel_number][idp]
                    free_channels[idp].append((mod_level, channel_number, idp))
        sorted_free_channels = [sorted(row, key=lambda x: (-x[0], x[1])) for row in free_channels]

        while True:
            max_modulation_level = float('-inf')
            max_modulation_row_index = None
            unassigned_bitrate = env.current_service.bit_rate
            selected_channels = []
            for i, row in enumerate(sorted_free_channels):
                # Check if the modulation level of the first tuple in the row is greater than or equal to the current maximum
                if row != [] and (row[0][0] > max_modulation_level or (
                        row[0][0] == max_modulation_level and i < max_modulation_row_index)):
                    max_modulation_level = row[0][0]
                    max_modulation_row_index = i

            if max_modulation_row_index is None:
                return (-2, [])
            else:
                for channel in sorted_free_channels[max_modulation_row_index]:
                    unassigned_bitrate -= int(channel[0]) * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((channel[1], channel[0] + unassigned_bitrate/100, unassigned_bitrate/-100, channel[0], False ))
                        return (channel[2], selected_channels)
                    else:
                        selected_channels.append((channel[1], channel[0], 0, channel[0], False))

                if len(sorted_free_channels) > max_modulation_row_index:
                    sorted_free_channels.pop(max_modulation_row_index)
                elif len(sorted_free_channels) == 0:
                    return (-2, [])
        return (-2, [])  # -2 means service is blocked


def phy_aware_bmfa_rmsa(env: PhyRMSAEnv) -> Tuple[int, list]:
    # first to check if we have enough resource in virtual layer
    if env.grooming:
        result = use_existing_channels(env)
    if env.grooming and result[0] != -3:
        return (result[0] + 20, result[1])
    else:
        table_id = np.where(((env.connections_detail[:, 0] == int(env.current_service.source)) & (
                    env.connections_detail[:, 1] == int(env.current_service.destination))) | (
                                    (env.connections_detail[:, 0] == int(env.current_service.destination)) & (
                                        env.connections_detail[:, 1] == int(env.current_service.source))))[0][0]
        free_channels = [[] for _ in range(env.k_paths)]

        ## this could be the Shortest available path first fit solution.
        for idp, path in enumerate(
                env.k_shortest_paths[
                    env.current_service.source, env.current_service.destination
                ]
        ):
            for channel_number in range(
                    0, env.topology.graph["num_channel_resources"]
            ):
                if env.is_channel_free(path, channel_number):
                    mod_level = env.modulation_level[table_id][channel_number][idp]
                    links_index = []
                    for i in range(len(path.node_list) - 1):
                        links_index.append(env.topology[path.node_list[i]][path.node_list[i + 1]]["index"])
                    fragmentation_metric = env.calculate_r_cut(channel_number, links_index, False, path, True)
                    free_channels[idp].append((mod_level, fragmentation_metric, channel_number, idp))

        sorted_free_channels = [sorted(row, key=lambda x: (-x[0], -x[1])) for row in free_channels]  ## this is for bmfa
        # sorted_free_channels = [sorted(row, key=lambda x: (-x[1], -x[0])) for row in free_channels]  ## this is for fabm

        while True:
            max_modulation_level = float('-inf')
            max_frag_metric = float('-inf')
            max_modulation_row_index = None
            unassigned_bitrate = env.current_service.bit_rate
            selected_channels = []
            for i, row in enumerate(sorted_free_channels):
                # Check if the modulation level of the first tuple in the row is greater than or equal to the current maximum
                if row != [] and (row[0][0] > max_modulation_level or (
                        row[0][0] == max_modulation_level and row[0][
                    1] > max_frag_metric)):  # it was i < max_modulation_row_index
                    max_modulation_level = row[0][0]
                    max_frag_metric = row[0][1]
                    max_modulation_row_index = i

            if max_modulation_row_index is None:
                return (-2, [])
            else:
                for channel in sorted_free_channels[max_modulation_row_index]:
                    unassigned_bitrate -= int(channel[0]) * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((channel[2], channel[0] + unassigned_bitrate/100, unassigned_bitrate/-100, channel[0], False ))
                        return (channel[3], selected_channels)
                    else:
                        selected_channels.append((channel[2], channel[0], 0, channel[0], False))

                if len(sorted_free_channels) > max_modulation_row_index:
                    sorted_free_channels.pop(max_modulation_row_index)
                elif len(sorted_free_channels) == 0:
                    return (-2, [])
        return (-2, [])


def phy_aware_bmfa_rss_rmsa(env: PhyRMSAEnv) -> Tuple[int, list]:
    # first to check if we have enough resource in virtual layer
    if env.grooming:
        result = use_existing_channels(env)
    if env.grooming and result[0] != -3:
        return (result[0] + 20, result[1])
    else:
        table_id = np.where(((env.connections_detail[:, 0] == int(env.current_service.source)) & (
                    env.connections_detail[:, 1] == int(env.current_service.destination))) | (
                                    (env.connections_detail[:, 0] == int(env.current_service.destination)) & (
                                        env.connections_detail[:, 1] == int(env.current_service.source))))[0][0]
        free_channels = [[] for _ in range(env.k_paths)]

        ## this could be the Shortest available path first fit solution.
        for idp, path in enumerate(
                env.k_shortest_paths[
                    env.current_service.source, env.current_service.destination
                ]
        ):
            for channel_number in range(
                    0, env.topology.graph["num_channel_resources"]
            ):
                if env.is_channel_free(path, channel_number):
                    mod_level = env.modulation_level[table_id][channel_number][idp]
                    links_index = []
                    for i in range(len(path.node_list) - 1):
                        links_index.append(env.topology[path.node_list[i]][path.node_list[i + 1]]["index"])
                    fragmentation_metric = env.calculate_r_spatial(channel_number, links_index)
                    free_channels[idp].append((mod_level, fragmentation_metric, channel_number, idp))

        sorted_free_channels = [sorted(row, key=lambda x: (-x[0], -x[1])) for row in free_channels]  ## this is for bmfa
        # sorted_free_channels = [sorted(row, key=lambda x: (-x[1], -x[0])) for row in free_channels]  ## this is for fabm

        while True:
            max_modulation_level = float('-inf')
            max_frag_metric = float('-inf')
            max_modulation_row_index = None
            unassigned_bitrate = env.current_service.bit_rate
            selected_channels = []
            for i, row in enumerate(sorted_free_channels):
                # Check if the modulation level of the first tuple in the row is greater than or equal to the current maximum
                if row != [] and (row[0][0] > max_modulation_level or (
                        row[0][0] == max_modulation_level and row[0][
                    1] > max_frag_metric)):  # it was i < max_modulation_row_index
                    max_modulation_level = row[0][0]
                    max_frag_metric = row[0][1]
                    max_modulation_row_index = i

            if max_modulation_row_index is None:
                return (-2, [])
            else:
                for channel in sorted_free_channels[max_modulation_row_index]:
                    unassigned_bitrate -= int(channel[0]) * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((channel[2], channel[0] + unassigned_bitrate / 100,
                                                  unassigned_bitrate / -100, channel[0], False))
                        return (channel[3], selected_channels)
                    else:
                        selected_channels.append((channel[2], channel[0], 0, channel[0], False))

                if len(sorted_free_channels) > max_modulation_row_index:
                    sorted_free_channels.pop(max_modulation_row_index)
                elif len(sorted_free_channels) == 0:
                    return (-2, [])
        return (-2, [])


def phy_aware_faff_rmsa(env: PhyRMSAEnv) -> Tuple[int, list]:
    # first to check if we have enough resource in virtual layer
    result = use_existing_channels(env)
    if result[0] != -3:
        return (result[0] + 20, result[1])
    else:
        table_id = np.where(((env.connections_detail[:, 0] == int(env.current_service.source)) & (
                    env.connections_detail[:, 1] == int(env.current_service.destination))) | (
                                    (env.connections_detail[:, 0] == int(env.current_service.destination)) & (
                                        env.connections_detail[:, 1] == int(env.current_service.source))))[0][0]
        free_channels = [[] for _ in range(env.k_paths)]

        ## this could be the Shortest available path first fit solution.
        for idp, path in enumerate(
                env.k_shortest_paths[
                    env.current_service.source, env.current_service.destination
                ]
        ):
            for channel_number in range(
                    0, env.topology.graph["num_channel_resources"]
            ):
                if env.is_channel_free(path, channel_number):
                    mod_level = env.modulation_level[table_id][channel_number][idp]
                    links_index = []
                    for i in range(len(path.node_list) - 1):
                        links_index.append(env.topology[path.node_list[i]][path.node_list[i + 1]]["index"])
                    fragmentation_metric = env.calculate_r_cut(channel_number, links_index, False, path, True)
                    free_channels[idp].append((mod_level, fragmentation_metric, channel_number, idp))

        # sorted_free_channels = [sorted(row, key=lambda x: (-x[0], -x[1])) for row in free_channels]  ## this is for bmfa
        sorted_free_channels = [sorted(row, key=lambda x: (-x[1])) for row in free_channels]  ## this is for fabm

        while True:
            max_modulation_level = float('-inf')
            max_frag_metric = float('-inf')
            max_modulation_row_index = None
            unassigned_bitrate = env.current_service.bit_rate
            selected_channels = []
            for i, row in enumerate(sorted_free_channels):
                # Check if the modulation level of the first tuple in the row is greater than or equal to the current maximum
                if row != [] and (row[0][1] > max_frag_metric):  # it was i < max_modulation_row_index
                    max_modulation_level = row[0][0]
                    max_frag_metric = row[0][1]
                    max_modulation_row_index = i

            if max_modulation_row_index is None:
                return (-2, [])
            else:
                for channel in sorted_free_channels[max_modulation_row_index]:
                    unassigned_bitrate -= int(channel[0]) * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((channel[2], channel[0] + unassigned_bitrate / 100,
                                                  unassigned_bitrate / -100, channel[0], False))
                        return (channel[3], selected_channels)
                    else:
                        selected_channels.append((channel[2], channel[0], 0, channel[0], False))

                if len(sorted_free_channels) > max_modulation_row_index:
                    sorted_free_channels.pop(max_modulation_row_index)
                elif len(sorted_free_channels) == 0:
                    return (-2, [])
        return (-2, [])


def phy_aware_faff_rss_rmsa(env: PhyRMSAEnv) -> Tuple[int, list]:
    # first to check if we have enough resource in virtual layer
    result = use_existing_channels(env)
    if result[0] != -3:
        return (result[0] + 20, result[1])
    else:
        table_id = np.where(((env.connections_detail[:, 0] == int(env.current_service.source)) & (
                    env.connections_detail[:, 1] == int(env.current_service.destination))) | (
                                    (env.connections_detail[:, 0] == int(env.current_service.destination)) & (
                                        env.connections_detail[:, 1] == int(env.current_service.source))))[0][0]
        free_channels = [[] for _ in range(env.k_paths)]

        ## this could be the Shortest available path first fit solution.
        for idp, path in enumerate(
                env.k_shortest_paths[
                    env.current_service.source, env.current_service.destination
                ]
        ):
            for channel_number in range(
                    0, env.topology.graph["num_channel_resources"]
            ):
                if env.is_channel_free(path, channel_number):
                    mod_level = env.modulation_level[table_id][channel_number][idp]
                    links_index = []
                    for i in range(len(path.node_list) - 1):
                        links_index.append(env.topology[path.node_list[i]][path.node_list[i + 1]]["index"])
                    fragmentation_metric = env.calculate_r_spatial(channel_number, links_index)
                    free_channels[idp].append((mod_level, fragmentation_metric, channel_number, idp))

        # sorted_free_channels = [sorted(row, key=lambda x: (-x[0], -x[1])) for row in free_channels]  ## this is for bmfa
        sorted_free_channels = [sorted(row, key=lambda x: (-x[1])) for row in free_channels]  ## this is for fabm

        while True:
            max_modulation_level = float('-inf')
            max_frag_metric = float('-inf')
            max_modulation_row_index = None
            unassigned_bitrate = env.current_service.bit_rate
            selected_channels = []
            for i, row in enumerate(sorted_free_channels):
                # Check if the modulation level of the first tuple in the row is greater than or equal to the current maximum
                if row != [] and (row[0][1] > max_frag_metric):  # it was i < max_modulation_row_index
                    max_modulation_level = row[0][0]
                    max_frag_metric = row[0][1]
                    max_modulation_row_index = i

            if max_modulation_row_index is None:
                return (-2, [])
            else:
                for channel in sorted_free_channels[max_modulation_row_index]:
                    unassigned_bitrate -= int(channel[0]) * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((channel[2], channel[0] + unassigned_bitrate / 100,
                                                  unassigned_bitrate / -100, channel[0], False))
                        return (channel[3], selected_channels)
                    else:
                        selected_channels.append((channel[2], channel[0], 0, channel[0], False))

                if len(sorted_free_channels) > max_modulation_row_index:
                    sorted_free_channels.pop(max_modulation_row_index)
                elif len(sorted_free_channels) == 0:
                    return (-2, [])
        return (-2, [])

# def shortest_available_path_first_fit(env: PhyRMSAEnv) -> Tuple[int, int]:
#     for idp, path in enumerate(
#             env.k_shortest_paths[
#                 env.current_service.source, env.current_service.destination
#             ]
#     ):
#         num_slots = env.get_number_slots(path)
#         for initial_slot in range(
#                 0, env.topology.graph["num_spectrum_resources"] - num_slots
#         ):
#             if env.is_path_free(path, initial_slot, num_slots):
#                 return (idp, initial_slot)
#     return (env.topology.graph["k_paths"], env.topology.graph["num_spectrum_resources"])


def use_existing_channels(env:PhyRMSAEnv) -> Tuple:

    unassigned_bitrate = env.current_service.bit_rate
    selected_channels = []

    for idp, path in enumerate(
            env.k_shortest_paths[
                env.current_service.source, env.current_service.destination
            ]
    ):
        if sum(candidate_channel[2] for candidate_channel in env.channel_state[env.current_service.source_id, env.current_service.destination_id, idp]) >= unassigned_bitrate/100:
            for candidate_channel in env.channel_state[env.current_service.source_id, env.current_service.destination_id, idp]:

                #Here you ca work with selected_channels  and pass it to the other function, or finish everyhing here.
                if candidate_channel[2] > 0:
                    unassigned_bitrate -= candidate_channel[2] * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((candidate_channel[0], candidate_channel[2] + unassigned_bitrate / 100,
                                                  unassigned_bitrate / -100, candidate_channel[3], True))
                        return (idp, selected_channels)
                    else:
                        selected_channels.append((candidate_channel[0], candidate_channel[2], 0, candidate_channel[3], True))

    return (-3, [])


def sapff_rmsa(env: PhyRMSAEnv) -> Tuple[int, list]:
    # first to check if we have enough resource in virtual layer
    result = use_existing_channels(env)
    if result[0] != -3:
        return (result[0] + 20, result[1])
    else:
        unassigned_bitrate = env.current_service.bit_rate
        selected_channels = []
        # free_channels = []
        table_id = np.where(((env.connections_detail[:, 0] == int(env.current_service.source)) & (
                env.connections_detail[:, 1] == int(env.current_service.destination))) | (
                                    (env.connections_detail[:, 0] == int(env.current_service.destination)) & (
                                    env.connections_detail[:, 1] == int(env.current_service.source))))[0][0]
        free_channels = [[] for _ in range(env.k_paths)]

        ## this could be the Shortest available path first fit solution.
        for idp, path in enumerate(
                env.k_shortest_paths[
                    env.current_service.source, env.current_service.destination
                ]
        ):
            for channel_number in range(
                    0, env.topology.graph["num_channel_resources"]
            ):
                if env.is_channel_free(path, channel_number):
                    mod_level = env.modulation_level[table_id][channel_number][idp]
                    free_channels[idp].append((mod_level, channel_number, idp))

        sorted_free_channels = [sorted(row, key=lambda x: (x[1])) for row in free_channels]

        while True:
            max_modulation_level = float('-inf')
            max_modulation_row_index = None
            unassigned_bitrate = env.current_service.bit_rate
            selected_channels = []
            for i in range(len(sorted_free_channels) - 1, -1, -1):
                # Check if the modulation level of the first tuple in the row is greater than or equal to the current maximum
                if sorted_free_channels[i] == []:
                    sorted_free_channels.pop(i)

            if len(sorted_free_channels) > 0:
                max_modulation_row_index = 0
                try:
                    max_modulation_level = sorted_free_channels[0][0][0]
                except:
                    aaa = 1
            if max_modulation_row_index is None:
                return (-2, [])
            else:
                for channel in sorted_free_channels[max_modulation_row_index]:
                    unassigned_bitrate -= int(channel[0]) * 100
                    if unassigned_bitrate <= 0:
                        selected_channels.append((channel[1], channel[0] + unassigned_bitrate / 100,
                                                  unassigned_bitrate / -100, channel[0], False))
                        return (channel[2], selected_channels)
                    else:
                        selected_channels.append((channel[1], channel[0], 0, channel[0], False))
                if len(sorted_free_channels) > max_modulation_row_index:
                    sorted_free_channels.pop(max_modulation_row_index)
                elif len(sorted_free_channels) == 0:
                    return (-2, [])
        return (-2, [])  # -2 means service is blocked

