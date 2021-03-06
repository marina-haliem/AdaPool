from collections import defaultdict
import numpy as np
from novelties import status_codes
from simulator.settings import FLAGS
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from simulator.models.customer.customer_repository import CustomerRepository
from common.geoutils import great_circle_distance
from novelties.pricing.price_calculator import calculate_price
from simulator.services.routing_service import RoutingEngine
from common import geoutils
import pandas as pd
import time as t
import operator
import concurrent.futures
import itertools
from functools import partial
from os import getpid

import multiprocessing as mp

class Central_Agent(object):

    def __init__(self, matching_policy):
        self.matching_policy = matching_policy
        self.routing_engine = RoutingEngine.create_engine()

    def getChunks(self, arr, n):
        avg = len(arr) / float(n)
        out = []
        last = 0.0

        while last < len(arr):
            out.append(arr[int(last) : int(last + avg)])
            last += avg

        return out

    def get_match_commands(self, current_time, vehicles, requests):
        # global VehicleRepository
        matching_commands = []
        num_unmatched = len(requests)
        finalize_commands = []
        all_matched_reqs = []
        total_matched = 0
        total_vehicles = pd.DataFrame(vehicles)
        numCpu = mp.cpu_count()
        # processMulti = int(numCpu*2.5)
        # cols = requests.columns
        # print("Cols: ", cols)
        if len(requests) > 0:
            total_reqs = len(requests)
            # print("R: ", len(requests))
            s = t.time()
            while(num_unmatched/total_reqs > 0.1):
                if FLAGS.enable_pooling:
                    s1 = t.time()
                    matching_commands = self.matching_policy.match(current_time, vehicles, requests)
                    e2 = t.time()
                    # print("Init Matching: ", e2-s1)
                # else:
                #     matching_commands = self.matching_policy.match(current_time, vehicles, requests)

                num_matched = len(matching_commands)
                # print("Matchings: ", num_matched)
                if num_matched == 0:
                    break

                matched_requests = [c["customer_id"] for c in matching_commands]
                # print("All_Requests: ", matched_requests)
                updated_commands = self.init_price(matching_commands)

                final_commands = self.consolidate(updated_commands)
                # print(len(final_commands), numCpu)
                # narrow_list, finalie, vehicles = self.greedy_insertion(vehicles, final_commands)

                # print(final_commands)
                print("111111111#####")
                print(len(final_commands))

                processMulti = int(len(final_commands)/30)
                # if len(final_commands) > 30:
                if(processMulti >= 2):
                    # print("Here")
                    s3 = t.time()
                    # print(numCpu, len(final_commands))
                    # items = final_commands
                    # chunksize = int((len(final_commands)+numCpu)/numCpu)
                    # chunksize = 10
                    # if (len(final_commands) - numCpu) > 10:
                    #     chunksize = int(len(final_commands) / numCpu)
                    # print(chunksize)
                    # chunks = []
            
                    # for i in range(0, len(items), chunksize):
                    #     if i + chunksize >= len(items):
                    #         chunks.append(items[i:len(items)])
                    #     else:
                    #         chunks.append(items[i:i + chunksize])

                    processMulti = min(processMulti, numCpu)
                    chunks = self.getChunks(final_commands, processMulti)
        
                    # # print(chunks)
                    e3 = t.time()
                    # print("Chunking: ", e3-s3)

                # reduced_values = p.map(func2, chunks)
                #     files = glob.iglob('*.csv')
                #     res = pool.map_async(partial(csv_to_df, dfs_list), files)
                #     res.wait()
                #     dfs = pd.concat(dfs_list, ignore_index=True)  # the final result
                #     print(dfs)

                    s4 = t.time()
                    # narrow_list, finalie = zip(*p.map(self.greedy_insertion, chunks))
                    narrow_list, finalie, final_dict = self.greedy_wrapper(chunks, processMulti)
                    e4 = t.time()
                    # p.close()
                    # print("Threading: ", e4 - s4)
                    # narrow_list.wait()
                    # print("Narrow Before: ", narrow_list)
                    # print(finalie, len(finalie))
                    # narrow_list = list(itertools.chain.from_iterable(narrow_list))
                    # finalie = list(itertools.chain.from_iterable(finalie))
                    # final_dict = list(itertools.chain.from_iterable(final_dict))
                    # print(final_dict)
                    # for c in final_dict:
                    #     v = VehicleRepository.get(c["vid"])
                    #     val= c["vals"]
                    #     v.current_plan = val[0]
                    #     v.ordered_pickups_dropoffs_ids = val[2]
                    #     v.pickup_flags = val[1]
                    #     v.current_plan_routes = val[3]
                    #     v.accepted_customers = val[4]
                    #     v.state.tmp_capacity = val[5]


                        # print(c["vehicle_id"], v.current_plan)
                        # print(v.ordered_pickups_dropoffs_ids, v.pickup_flags)

                    # print(list(itertools.chain.from_iterable(narrow_list)))
                    # print(list(itertools.chain.from_iterable(finalie)))
                    # print("Narrow After: ", narrow_list)
                    # print(finalie, len(finalie))
                    # narrow_list, finalie = Result

                    # p.join()
                else:
                    # print("There")
                    narrow_list, finalie, final_dict = self.greedy_insertion(commands = final_commands)

                # print(vehicles)
                for index, v in vehicles.iterrows():
                    # print(index)
                    vehicle = VehicleRepository.get(index)
                    if vehicle.state.tmp_capacity >= vehicle.state.max_capacity:
                        vehicles.drop(index, inplace=True)
                        vehicle.tmp_capacity = vehicle.state.current_capacity
                # narrow_list, finalie = (zip*[p.map(self.greedy_insertion, final_commands)])

                # executor = concurrent.futures.ProcessPoolExecutor(10)
                # narrow_list, finalie, vehicles = [executor.submit(self.greedy_insertion, vehicles, group)
                #            for group in more_itertools.grouper(5, final_commands)]
                # concurrent.futures.wait(narrow_list, finalie, vehicles)

                # print("Matched: ", narrow_list)
                # print("Finalie: ", finalie)
                finalize_commands.extend(finalie)
                # print(len(finalie), len(finalize_commands))
                num_unmatched = num_matched - len(narrow_list)
                total_matched += len(narrow_list)
                all_matched_reqs.extend(narrow_list)
                if num_unmatched/num_matched < 0.1:
                    break
                # matched_requests.remove(narrow_list)
                request_ids = set(matched_requests).difference(narrow_list)
                # print("Unmatched: ", request_ids)

                reqs = [CustomerRepository.get(i).get_request() for i in request_ids]
                cols = CustomerRepository.get_col_names()
                # print("Cols: ", cols)
                requests = pd.DataFrame.from_records(reqs, columns=cols)
                requests = requests.set_index("id")

            # print(len(finalize_commands), total_matched)
            finalize_commands = self.consolidate(finalize_commands)
            vehicles = self.update_vehicles(total_vehicles, finalize_commands)
            # print("Finallllllllllly: ", [[c["vehicle_id"], c["customer_id"]] for c in finalize_commands])
            for c in finalize_commands:
                for id in range(len(c["customer_id"])):
                    cust = CustomerRepository.get(c["customer_id"][id])
                    if cust is None:
                        print("Invalid Customer")
                    cust.wait_for_vehicle(c["duration"][id])
            # for c in finalize_commands:
            #     v = VehicleRepository.get(c["vehicle_id"])
            #     print(c["vehicle_id"], v.current_plan)
            #     print(v.ordered_pickups_dropoffs_ids, v.pickup_flags)
            end = t.time()
            print("Matching Time: ", end-s, "Rate: ", (total_matched/total_reqs)*100, "%")
        else:
            return [], vehicles
        return finalize_commands, vehicles, total_matched


    def update_vehicles(self, vehicles, commands):
        vehicle_ids = [command["vehicle_id"] for command in commands]
        vehicles.loc[vehicle_ids, "status"] = status_codes.V_ASSIGNED
        # print(vehicles.loc[vehicle_ids,"status"])
        return vehicles

    def consolidate(self, commands):
        final_commands = []
        if FLAGS.enable_pooling:

            V = defaultdict(list)
            # V_pair = defaultdict(list)
            V_duration = defaultdict(list)
            V_price = defaultdict(list)
            # V_dist = defaultdict(list)
            # V_dest = defaultdict(list)

            for command in commands:
                # if command["vehicle_id"] in V.keys():
                #
                # else:
                if type(command["customer_id"]) is not list:
                    V[command["vehicle_id"]].append(command["customer_id"])
                    V_duration[command["vehicle_id"]].append(command["duration"])
                    V_price[command["vehicle_id"]].append(command["init_price"])
                else:
                    V[command["vehicle_id"]].extend(list(command["customer_id"]))
                    V_duration[command["vehicle_id"]].extend(list(command["duration"]))
                    V_price[command["vehicle_id"]].extend(list(command["init_price"]))
                    # V_dest[command["vehicle_id"]].append(customer.get_destination())
                    # V_dist[command["vehicle_id"]].append(command["distance"])
                # customer = CustomerRepository.get(command["customer_id"])

            for k in V.keys():
                # print(k)
                new_command = dict()
                new_command["vehicle_id"] = k
                new_command["customer_id"] = V[k]
                # new_command["pickups"] = V_pair[k]
                new_command["duration"] = V_duration[k]
                # new_command["distance"] = V_dist[k]
                new_command["init_price"] = V_price[k]
                # new_command["destinations"] = V_dest[k]
                final_commands.append(new_command)

        else:
            final_commands = commands

        return final_commands

    def init_price(self, match_commands):
        m_commands = []
        for c in match_commands:
            vehicle = VehicleRepository.get(c["vehicle_id"])
            # vehicle.state.status = status_codes.V_ASSIGNED
            if vehicle is None:
                print("Invalid Vehicle id")
                continue
            vehicle.state.init_match_count = 0
            customer = CustomerRepository.get(c["customer_id"])
            if customer is None:
                print("Invalid Customer id")
                continue

            # triptime = c["duration"]

            # if FLAGS.enable_pricing:
            # dist_for_pickup = c["distance"]

            # od_route = self.routing_engine.route_time([(customer.get_origin(), customer.get_destination())])
            # # dist_dropoff = great_circle_distance(customer.get_origin()[0], customer.get_origin()[1],
            # #                               customer.get_destination()[0], customer.get_destination()[1])
            # route, time = od_route[0]
            # lats, lons = zip(*route)
            # distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])  # Distance in meters
            # dist_till_dropoff = sum(distance)
            #
            # # print(dist_dropoff, dist_till_dropoff)
            #
            # total_trip_dist = dist_till_dropoff
            # [travel_price, wait_price] = vehicle.get_price_rates()
            # # v_cap = vehicle.state.current_capacity
            # initial_price = calculate_price(total_trip_dist, triptime, vehicle.state.mileage, travel_price,
            #                     wait_price, vehicle.state.gas_price, vehicle.state.driver_base_per_trip)
            c["init_price"] = customer.get_request().fare
            m_commands.append(c)
            # print(c)

        return m_commands


    def generate_plan(self, vehicle, cust_list, new_customer):
        # global VehicleRepository
        # print("In Plan: ", vehicle.get_id(), vehicle.current_plan)
        # print(vehicle.pickup_flags, vehicle.ordered_pickups_dropoffs_ids)
        s = t.time()
        tmp_plan = np.copy(vehicle.current_plan).tolist()
        tmp_flags = np.copy(vehicle.pickup_flags).tolist()
        tmp_ids = np.copy(vehicle.ordered_pickups_dropoffs_ids).tolist()
        tmp_routes = np.copy(vehicle.current_plan_routes).tolist()

        time_till_pickup = 0
        insertion_cost = 0
        # distance_till_dropoff = 0
        if len(cust_list) == 0 & len(tmp_plan) == 0:
            tmp_plan.append(new_customer.get_origin())
            tmp_plan.append(new_customer.get_destination())
            tmp_ids.append(new_customer.get_id())
            tmp_ids.append(new_customer.get_id())
            tmp_flags.append(1)
            tmp_flags.append(0)

            routes_till_pickup = [(vehicle.get_location(), tmp_plan[0])]
            routes = self.routing_engine.route_time(routes_till_pickup)

            for (route, time) in routes:
                time_till_pickup += time

            od_pairs = [(vehicle.get_location(), tmp_plan[0]),
                        (tmp_plan[0], tmp_plan[1])]
            routes = self.routing_engine.route_time(od_pairs)
            for (route, time) in routes:
                tmp_routes.append([route, time])
                lats, lons = zip(*route)
                distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],
                                                          lons[1:])  # Distance in meters
                insertion_cost += sum(distance)

            vehicle_plan = np.copy(tmp_plan)
            vehicle_flags = np.copy(tmp_flags)
            vehicle_ids = np.copy(tmp_ids)
            vehicle_routes = np.copy(tmp_routes)

        else:
            new_list = [new_customer.get_origin(), new_customer.get_destination()]
            # print(new_list)
            final_pickup_flags = []
            final_plan = []
            final_pickups_dropoffs_ids = []
            final_routes = []

            min_distance = np.inf
            min_time = np.inf
            pickup_index = 0
            # all_options = vehicle.current_plan + new_list
            # print(all_options)
            # perm = list(itertools.permutations(all_options))
            # print(perm)
            # print(len(set(perm)))
            # print("Insert Pickup")
            for pos in range(len(vehicle.current_plan)+1):
                # print(vehicle.current_plan)
                new_plan = np.copy(vehicle.current_plan[:pos]).tolist()
                new_pickup_flags = np.copy(vehicle.pickup_flags[:pos]).tolist()
                new_pickups_dropoffs_ids = np.copy(vehicle.ordered_pickups_dropoffs_ids[:pos]).tolist()
                new_plan.append(new_list[0])
                new_pickup_flags.append(1)
                new_pickups_dropoffs_ids.append(new_customer.get_id())
                new_plan.extend(vehicle.current_plan[pos:])
                new_pickup_flags.extend(vehicle.pickup_flags[pos:])
                new_pickups_dropoffs_ids.extend(vehicle.ordered_pickups_dropoffs_ids[pos:])

                # print("List: ", new_plan, new_pickup_flags, new_pickups_dropoffs_ids)
                od_pairs = [(vehicle.get_location(), new_plan[0])]
                od_pairs.extend([(new_plan[x], new_plan[x + 1]) for x in range(len(new_plan) - 1)])
                # print(od_pairs)
                total_time = 0
                total_dist = 0
                potential_routes_time = self.routing_engine.route_time(od_pairs)
                new_routes = []
                index = 0
                wait_time = 0
                # pickup_distance = 0
                for (route, time) in potential_routes_time:
                    total_time += time
                    lats, lons = zip(*route)
                    distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],lons[1:])  # Distance in meters
                    total_dist += sum(distance)
                    new_routes.append([route, time])
                    if index < pos+1:
                        wait_time += time
                        # pickup_distance += sum(distance)
                    index += 1
                # print("Pos: ", pos, "Wait: ", wait_time)
                # print("T: ", total_time)
                # print("D: ", total_dist)

                if total_dist < min_distance:
                    min_time = total_time
                    min_distance = total_dist
                    insertion_cost = total_dist
                    final_pickup_flags = new_pickup_flags
                    final_plan = new_plan
                    final_pickups_dropoffs_ids = new_pickups_dropoffs_ids
                    pickup_index = pos
                    final_routes = new_routes
                    time_till_pickup = wait_time
                    # distance_till_pickup = pickup_distance

                # print("Min: ", min_time, min_distance)
            tmp_plan = np.copy(final_plan)
            tmp_flags = np.copy(final_pickup_flags)
            tmp_ids = np.copy(final_pickups_dropoffs_ids)
            tmp_routes = np.copy(final_routes)

            final_pickup_flags = []
            final_plan = []
            final_pickups_dropoffs_ids = []
            final_routes = []

            min_distance = np.inf
            min_time = np.inf

            # print("Insert Drop-off! ", pickup_index, time_till_pickup, vehicle.current_plan)

            for pos in range(len(tmp_plan)+1):
                if pos <= pickup_index:
                    continue
                # print(vehicle.current_plan)
                new_plan = np.copy(tmp_plan[:pos]).tolist()
                new_pickup_flags = np.copy(tmp_flags[:pos]).tolist()
                new_pickups_dropoffs_ids = np.copy(tmp_ids[:pos]).tolist()
                new_plan.append(new_list[1])
                new_pickup_flags.append(0)
                new_pickups_dropoffs_ids.append(new_customer.get_id())
                new_plan.extend(tmp_plan[pos:])
                new_pickup_flags.extend(tmp_flags[pos:])
                new_pickups_dropoffs_ids.extend(tmp_ids[pos:])

                # print("List: ", new_plan, new_pickup_flags, new_pickups_dropoffs_ids)
                od_pairs = [(vehicle.get_location(), new_plan[0])]
                od_pairs.extend([(new_plan[x], new_plan[x+1]) for x in range(len(new_plan)-1)])
                total_time = 0
                total_dist = 0
                potential_routes_time = self.routing_engine.route_time(od_pairs)
                # print(len(new_plan)+1, len(od_pairs), len(potential_routes_time))
                new_routes = []
                # counter = 0
                # dropoff_distance = 0
                for (route, time) in potential_routes_time:
                    total_time += time
                    lats, lons = zip(*route)
                    distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],lons[1:])  # Distance in meters
                    total_dist += sum(distance)
                    new_routes.append([route, time])
                    # if pickup_index < counter < pos+1:
                    #     dropoff_distance += sum(distance)
                    # counter += 1
                # print("T: ", total_time)
                # print("D: ", total_dist)

                if total_dist < min_distance:
                    min_time = total_time
                    min_distance = total_dist
                    insertion_cost = total_dist
                    final_pickup_flags = np.copy(new_pickup_flags).tolist()
                    final_plan = np.copy(new_plan).tolist()
                    final_pickups_dropoffs_ids = np.copy(new_pickups_dropoffs_ids).tolist()
                    final_routes = np.copy(new_routes)
                    # distance_till_dropoff = dropoff_distance

                # print("Min: ", min_time, min_distance)

            vehicle_plan = np.copy(final_plan).tolist()
            vehicle_flags = np.copy(final_pickup_flags).tolist()
            vehicle_ids = np.copy(final_pickups_dropoffs_ids).tolist()
            vehicle_routes = np.copy(final_routes).tolist()

            # print("Generated Plan: ", vehicle.current_plan)
            # print("For vid: ", vehicle.get_id(), "and cust: ", new_customer.get_id())
            # print(vehicle_flags, vehicle_ids)
            # dist_time = []
            # for (route, time) in vehicle.current_plan_routes:
            #     lats, lons = zip(*route)
            #     distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],lons[1:])  # Distance in meters
            #     dist = sum(distance)
            #     dist_time.append([dist, time])
            # print("Routes: ", dist_time)

            # print(time_till_pickup, distance_till_pickup, distance_till_dropoff)
            # print("Nxt!!")
        e = t.time()
        # print("Generate Plan: ", e-s)
        return [insertion_cost, time_till_pickup,  vehicle_plan, vehicle_flags, vehicle_ids, vehicle_routes]


    def greedy_insertion(self, commands, result_queue = None):
        # from simulator.models.vehicle.vehicle_repository import VehicleRepository
        # global VehicleRepository
        # print("Here")
        print(mp.current_process(), getpid())
        s = t.time()
        vehicle_accepted_cust = defaultdict()
        matched_requests = []
        final_commands = []

        # print(commands)
        c = []
        if type(commands) != list:
            # commands = list(commands)
            c.append(commands)
            commands = c
        # print(commands)
        final_dict = []
        for command in commands:
            # print("inside For", command)
            rejected_flag = 0
            # print(command["vehicle_id"], command["customer_id"])
            vehicle = VehicleRepository.get(command["vehicle_id"])
            vid = command["vehicle_id"]
            # print(VehicleRepository.get(vid))
            # print("V_Loc: ", vehicle.get_location())
            # vehicle.state.status = status_codes.V_ASSIGNED
            if vehicle is None:
                # self.logger.warning("Invalid Vehicle id")
                # print("Error")
                continue
            # print("Vid: ", vid, "Plan: ", vehicle.current_plan)
            # vehicle_cust_price_time = dict()

            if (vehicle.state.status == status_codes.V_OCCUPIED) & vehicle.state.accept_new_request:
                # print(vid, "Update", vehicle.current_plan, vehicle.get_destination())
                if (len(vehicle.current_plan) == 0) & (vehicle.state.destination_lat is not None):
                    # print(vid, "Dest: ", vehicle.get_destination())
                    vehicle.current_plan = [vehicle.get_destination()]
                    vehicle.current_plan_routes = [(vehicle.get_route(), vehicle.state.time_to_destination)]

                elif (len(vehicle.current_plan) == 0) & (vehicle.state.destination_lat is None):
                    vehicle.change_to_idle()
                    vehicle.reset_plan()

                elif (len(vehicle.current_plan) != 0) & (
                        vehicle.get_destination() != vehicle.current_plan[0]):
                    # print(vid, ": ", vehicle.get_destination(), vehicle.current_plan[0])
                    if vehicle.state.destination_lat is not None:
                        plan = [vehicle.get_destination()]
                        plan.extend(vehicle.current_plan)
                        vehicle.current_plan = np.copy(plan).tolist()

                        if len(vehicle.get_route()) == 0:
                            # print(vid, "Empty Route!!", vehicle.get_destination())
                            # print(vehicle.current_plan)
                            # print(vehicle.pickup_flags)
                            # print(vehicle.ordered_pickups_dropoffs_ids)
                            # print(vehicle.current_plan_routes)

                            od_routes = self.routing_engine.route_time([(vehicle.get_location(),
                                                                     vehicle.get_destination())])
                            routes = [od_routes[0]]
                        else:

                            routes = [[vehicle.get_route(), vehicle.state.time_to_destination]]

                        routes.extend(vehicle.current_plan_routes)
                        vehicle.current_plan_routes = np.copy(routes).tolist()

                        # if len(vehicle.get_route()) == 0:
                        #     print("R: ", vehicle.current_plan_routes)

                if len(vehicle.current_plan) != len(vehicle.current_plan_routes) != len(\
                        vehicle.ordered_pickups_dropoffs_ids) != len(vehicle.pickup_flags):
                    print("ERROR!")

            # print("Before While")
            vehicle_accepted_cust[vid] = list()
            # print("vid: ", vid)
            # print("Cust_List: ", list(command["customer_id"]))
            # vehicle.tmp_capacity = vehicle.state.current_capacity
            # numcpu = multiprocessing.cpu_count()
            # print(vid, vehicle.state.tmp_capacity, vehicle.state.max_capacity)
            while vehicle.state.tmp_capacity < vehicle.state.max_capacity:
                # print("Inside While")
                insertion_cost_dict = dict()
                cust_index_dict = dict()

                for index in range(len(command["customer_id"])):
                # customer_dict = {index:CustomerRepository.get(command["customer_id"][index]) for index in range(len(
                #         command["customer_id"]))}
                    customer = CustomerRepository.get(command["customer_id"][index])
                    if customer is None:
                        print("Invalid Customer id")
                        continue

                # p = multiprocessing.Pool(numcpu)
                # insertion_cost_dict, cust_index_dict = (zip* p.map(partial(self.generate_plan,
                #                                                            customer=customer_dict),vehicle,
                #                                                    vehicle_accepted_cust[vid]))

                    cid = command["customer_id"][index]

                    [insertion_cost, waiting_time, Vplan, Vflags, Vids, Vroutes] = self.generate_plan(vehicle,
                                                                                          vehicle_accepted_cust[vid], customer)
                    insertion_cost_dict[cid] = insertion_cost
                    price = command["init_price"][index]
                    cust_index_dict[cid] = [index, waiting_time, price, Vplan, Vflags, Vids, Vroutes]
                    command["duration"][index] = waiting_time

                    # if len(vehicle_accepted_cust[vid]) > 1:
                    #     # print("prev: ", prev_cost, insertion_cost)
                    #     insertion_cost = abs(insertion_cost - prev_cost)
                    #
                    # if insertion_cost == float(0):
                    #     insertion_cost = command["init_price"][index]
                    # print("L: ", len(vehicle_accepted_cust[vid]))
                    # print("C: ", insertion_cost)

                    # saved_time = command["duration"][index] - waiting_time
                    # [travel_price, wait_price] = vehicle.get_price_rates()
                    # # init_price += saved_time*wait_price
                    #
                    # init_price = calculate_price(insertion_cost, waiting_time, vehicle.state.mileage, travel_price,
                    #                                 wait_price, vehicle.state.gas_price,
                    #                                 vehicle.state.driver_base_per_trip)

                    # print("P: ", init_price, command["init_price"][index])


                    # command["init_price"][index] = customer.get_request().fare
                e = t.time()
                # print("Dict: ", insertion_cost_dict)
                # min_cost = min(insertion_cost_dict.values())
                # min_request_id = [k for k in insertion_cost_dict.keys() if insertion_cost_dict[k] == min_cost][0]
                min_request_id = min(insertion_cost_dict.items(), key=operator.itemgetter(1))[0]
                num, wait, price, Min_plan, Min_flags, Min_ids, Min_routes = np.copy(cust_index_dict[min_request_id])
                # print("min_cost:", min_request_id)
                # print(Min_plan, Min_flags, Min_ids)
                matched_requests.append(min_request_id)
                vehicle.accepted_customers.append(min_request_id)
                vehicle_accepted_cust[vid].append([min_request_id, wait, price])
                vehicle.state.tmp_capacity += 1
                # cust = CustomerRepository.get(command["customer_id"][num])
                # cust.wait_for_vehicle(command["duration"][num])
                command["customer_id"].remove(min_request_id)
                command["duration"].remove(wait)
                # print("Cust_after: ", command["customer_id"])
                # print("Duration after: ", command["duration"])
                # prev_cost = min_cost

                vehicle.current_plan = np.copy(Min_plan).tolist()
                vehicle.pickup_flags = np.copy(Min_flags).tolist()
                vehicle.ordered_pickups_dropoffs_ids = np.copy(Min_ids).tolist()
                vehicle.current_plan_routes = np.copy(Min_routes).tolist()

                if len(list(command["customer_id"])) == 0:
                    break

            cust_id_wait = np.array(vehicle_accepted_cust[vid])
            # print("Acc: ", vehicle.state.tmp_capacity, vehicle.state.max_capacity)
            if len(cust_id_wait) != 0:
                # print("Acc: ", cust_id_wait[:,0])
                command["customer_id"] = cust_id_wait[:, 0].tolist()
                command["duration"] = cust_id_wait[:, 1].tolist()
                command["init_price"] = cust_id_wait[:, 2].tolist()
                # print(command)
                final_commands.append(command)
            else:
                command["customer_id"] = list()
                command["duration"] = list()
                command["init_price"] = list()

            # print("Final Plan: ", vid)
            # print(vehicle.current_plan)
            # vehicle.current_plan = np.copy(vehicle.current_plan).tolist()
            # vehicle.pickup_flags = np.copy(vehicle.pickup_flags).tolist()
            # vehicle.ordered_pickups_dropoffs_ids = np.copy(vehicle.ordered_pickups_dropoffs_ids).tolist()
            # vehicle.current_plan_routes = np.copy(vehicle.current_plan_routes).tolist()

            tmp_dict = dict()
            tmp_dict["vid"] = vid
            tmp_dict["vals"] = [vehicle.current_plan, vehicle.pickup_flags, vehicle.ordered_pickups_dropoffs_ids,
                                vehicle.current_plan_routes, vehicle.accepted_customers, vehicle.state.tmp_capacity]
            final_dict.append(tmp_dict)
            # print(vehicle.pickup_flags)
            # print(vehicle.ordered_pickups_dropoffs_ids )
            # if vehicle.state.tmp_capacity >= vehicle.state.max_capacity:
            #     vehicles = vehicles.drop(vid)
            # vehicle.tmp_capacity = vehicle.state.current_capacity

        print("Greedy Insertion: ", e-s, len(commands))
        # print("Greedy Insertion Done")
        # print(matched_requests, final_commands)

        if(result_queue != None):
            result_queue.put((matched_requests, final_commands, final_dict))
        else:
            return (matched_requests, final_commands, final_dict)


    # def mergePoolMap(self, poolMap):
    #     narrow = []
    #     commands = []
    #     final_dict = [{}]
    #
    #     for mapE in poolMap:
    #         narrow += mapE[0]
    #         commands += mapE[1]
    #         final_dict += mapE[2]
    #
    #     return (narrow, commands, final_dict)

    def greedy_wrapper(self, chunks, processMulti):
        # numCpu = multiprocessing.cpu_count()
        # p = multiprocessing.Pool(processes=5)
        # chunks = list()
        # with multiprocessing.Pool(numCpu) as p:
        # size = 1
        # if (len(chunks) - numCpu) > 10:
        #     size = int(len(chunks)/numCpu)
        # ctx = mp.get_context("spawn")
        
        # p = ctx.Pool(processes=processMulti)
        # poolMap = p.map(self.greedy_insertion, chunks)
        # resultSet = self.mergePoolMap(poolMap)
        # p.close()
        # p.terminate()

        processList = []
        result_queue = mp.Queue()

        for c in chunks:
            p = mp.Process(target=self.greedy_insertion, args=(c, result_queue,))
            processList.append(p)
            p.start()

        matched_requests = []
        final_commands = []
        final_dict = []

        for p in processList:
            result = result_queue.get()

            matched_requests += result[0]
            final_commands += result[1]
            final_dict += result[2]


        return (matched_requests, final_commands, final_dict)