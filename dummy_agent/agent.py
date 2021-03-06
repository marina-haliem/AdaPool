from simulator.settings import FLAGS, INITIAL_MEMORY_SIZE
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from simulator.settings import FLAGS
from logger import agent_logger
from logging import getLogger
import pandas as pd
import numpy as np
import dirichlet.dirichlet
from sklearn import preprocessing
import math
import operator
import time
import multiprocessing as mp

agent_log_cols = ["t", "active_model", "current_state_lat", "current_state_lon", "reward", "next_state_lat",
                  "next_state_lon"]

last_change = FLAGS.start_time

class Dummy_Agent(object):

    def __init__(self, pricing_policy, dispatch_policy):
        self.pricing_policy = pricing_policy
        if FLAGS.adaptive:
            self.models = dispatch_policy
        else:
            self.dispatch_policy = dispatch_policy

    def get_dispatch_commands(self, current_time, vehicles):

        # dispatch_commands = self.dispatch_policy.dispatch(current_time, vehicles)
        # return dispatch_commands
        return []

    # def get_price_decision(self, vehicle, price, request):
    #     response = self.pricing_policy.propose_price(vehicle, price, request)
    #
    #     return response

class DQN_Agent(Dummy_Agent):

    def __init__(self, pricing_policy, dispatch_policy, start_time=None, timestep=None):
        super().__init__(pricing_policy, dispatch_policy)
        if start_time is not None:
            self.__t = start_time
        if timestep is not None:
            self.__dt = timestep
        agent_logger.setup_logging(self)
        self.logger = getLogger(__name__)
        self.model_num = 0
        self.dirichlet_threshold = 6500
        if FLAGS.adaptive:

            self.current_model = self.models[0]

    def get_dispatch_commands(self, current_time, vehicles):
        path = "./logs/tmp/sim/models.log"
        global last_change
        if FLAGS.adaptive:
            dispatch_commands = self.current_model.dispatch(current_time, vehicles, agent_logger, self.model_num)
        else:
            dispatch_commands = self.dispatch_policy.dispatch(current_time, vehicles, agent_logger, self.model_num)

        if FLAGS.adaptive:
            df = pd.read_csv(path, names=agent_log_cols)
            df = df[df.active_model == self.model_num]
            # print(agent_log_cols)
            t_values = df['t'].values
            # print(len(np.unique(t_values)))
            df = df.drop(['t'], axis=1)
            if len(df) > 0:
                # Data Normalization
                x = df.values  # returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.0000001, 1))
                x_scaled = min_max_scaler.fit_transform(x)
                df = pd.DataFrame(x_scaled)
                df.columns = agent_log_cols[1:]
                df['t'] = t_values
                df = df.drop(['active_model'], axis=1)
                # Get Data after last change, and split it at every t to decide on best split
                df_right = df[df.t >= last_change]      # Get Data after last change
                # print("Change: ", last_change, "Current_time: ", current_time, len(df_right))
                if len(df_right) > 10 and (current_time - last_change) >= (5*60): # After 5 minutes of last change
                    # print("test")
                    self.change_point_detection(df_right, current_time)

        self.__update_time()
        return dispatch_commands

    def getChunks(self, arr, n):
        avg = len(arr) / float(n)
        out = []
        last = 0.0

        while last < len(arr):
            out.append(arr[int(last) : int(last + avg)])
            last += avg
        return out

    def change_point_detection(self, all_data, current_time):
        global last_change
        ll_dict = dict()
        t_values = all_data['t'].values
        # print(last_change, len(np.unique(t_values)))
        s = time.time()
        times = np.unique(t_values)

        numCpu = mp.cpu_count()
        processMulti = int(len(times) / 30)
        # if len(final_commands) > 30:
        if (processMulti >= 2):
            processMulti = min(processMulti, numCpu)
            time_chunks = self.getChunks(times, processMulti)
        # time_chunks = self.getChunks(times, len(times))
        #     print(len(time_chunks))
        # for t in np.unique(t_values):
        #     right_data = all_data[all_data.t > t]
        #     right_data = right_data.drop(['t'], axis=1)
        #     # print(df_2, len(df_2), np.log(df_2).mean())
        #     left_data = all_data[all_data.t <= t]
        #     left_data = left_data.drop(['t'], axis=1)
        #     # print(df_1, len(df_1), np.log(df_1).mean())
        #     # print(len(right_data), len(left_data))
        #     if len(right_data) > 1 and len(left_data) > 1:
        #         ll = dirichlet.test(np.array(left_data), np.array(right_data), method="meanprecision", maxiter=1000)
        #         ll_dict[t] = ll

            ll_dict = self.change_wrapper(all_data, time_chunks)
        else:
            for t in np.unique(t_values):
                right_data = all_data[all_data.t > t]
                right_data = right_data.drop(['t'], axis=1)
                # print(df_2, len(df_2), np.log(df_2).mean())
                left_data = all_data[all_data.t <= t]
                left_data = left_data.drop(['t'], axis=1)
                # print(df_1, len(df_1), np.log(df_1).mean())
                # print(len(right_data), len(left_data))
                if len(right_data) > 1 and len(left_data) > 1:
                    ll = dirichlet.test(np.array(left_data), np.array(right_data), method="meanprecision", maxiter=1000)
                    ll_dict[t] = ll
        # print("Times: ", len(ll_dict), ll_dict)
        if len(ll_dict) > 0:
            # print(ll_dict)
            max_ll = max(ll_dict.values())
            max_t = max(ll_dict.items(), key=operator.itemgetter(1))[0]
            # print(max_t, max_ll)
            all_data = all_data.drop(['t'], axis=1)
            # print("All: ", np.array(all_data))
            # D0 = vstack((np.array(left_data), np.array(right_data)))
            D0 = np.array(all_data)
            # print("D0: ", D0)
            a0 = dirichlet.mle(D0, method="meanprecision", maxiter=1000)
            ll0 = dirichlet.loglikelihood(D0, a0)
            pval = max_ll-ll0
            # print("LL0: ", ll0, "LL* - LL0: ", pval)
            if pval > self.dirichlet_threshold:
                print("Dirichlet: ", pval)
                print("Switch at: ", max_t)
                self.model_num = int(math.fmod(self.model_num+1, len(self.models)))
                self.current_model = self.models[self.model_num]
                last_change = max_t

        end = time.time()
        # print("Change Detection Time: ", end - s)

    def get_change_data(self, all_data, point, result_queue):
        # print("p: ", len(point))
        outt_dict = dict()
        for t in point:
            # print(t)
            right_data = all_data[all_data.t > t]
            right_data = right_data.drop(['t'], axis=1)
            # print(df_2, len(df_2), np.log(df_2).mean())
            left_data = all_data[all_data.t <= t]
            left_data = left_data.drop(['t'], axis=1)
            # print(df_1, len(df_1), np.log(df_1).mean())
            # print(len(right_data), len(left_data))
            if len(right_data) > 1 and len(left_data) > 1:
                ll = dirichlet.test(np.array(left_data), np.array(right_data), method="meanprecision", maxiter=1000)
                outt_dict[t] = ll
            # print(outt_dict)
        #     if (result_queue != None):
        #         result_queue.put((outt_dict))
        #     else:
        #         return outt_dict
        # else:
            # print(t, ": Here")
        if (result_queue != None):
            result_queue.put((outt_dict))
        else:
            return outt_dict

    def change_wrapper(self, all_data, chunks):
        processList = []
        res_queue = mp.Queue()
        # print(len(chunks))
        for c in chunks:
            p = mp.Process(target=self.get_change_data, args=(all_data, c, res_queue,))
            processList.append(p)
            p.start()

        outt_dict = dict()
        # print("Out: ", len(processList))
        for p in processList:
            result = res_queue.get()
            # print("Res: ", result)
            if result != None:
                outt_dict.update(result)
            # outt_dict[point] = result
            # final_commands += result[1]
            # final_dict += result[2]
        # print("Done!")
        return outt_dict

    def get_current_time(self):
        t = self.__t
        return t

    def __update_time(self):
        self.__t += self.__dt

    # def startup_dispatch(self, current_time, vehicles):
    #     self.dispatch_policy.update_state(current_time, vehicles)
    #     if FLAGS.train:
    #         self.dispatch_policy.give_rewards(vehicles)
    #     dispatch_commands = self.dispatch_policy.get_dispatch_decisions(vehicles)
    #     self.dispatch_policy.record_dispatch(vehicles.index, current_time)
    #     for vid in vehicles.index:
    #         vehicle = VehicleRepository.get(vid)
    #         vehicle.first_dispatched = 1
    #     ######### Only with the DQN agent not the dummy #############
    #     if FLAGS.train:
    #         self.dispatch_policy.backup_supply_demand()
    #         # If size exceeded, run training
    #         if len(self.dispatch_policy.supply_demand_history) > INITIAL_MEMORY_SIZE:
    #             average_loss, average_q_max = self.dispatch_policy.train_network(FLAGS.batch_size)
    #             # print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(
    #             #     self.q_network.n_steps, average_loss, average_q_max), flush=True)
    #             self.dispatch_policy.q_network.write_summary(average_loss, average_q_max)
    #     return dispatch_commands
    #
    #
