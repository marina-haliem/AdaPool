from collections import defaultdict

import numpy as np

from common import geoutils
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from simulator.models.customer.customer_repository import CustomerRepository
from simulator.services.demand_generation_service import DemandGenerator
from simulator.services.routing_service import RoutingEngine
from common.time_utils import get_local_datetime
from config.settings import OFF_DURATION, PICKUP_DURATION
from simulator.settings import FLAGS
from logger import sim_logger
from logging import getLogger
from novelties import agent_codes, status_codes
# from novelties.pricing.price_calculator import recalculate_price_per_customer
from random import randrange
from common.geoutils import great_circle_distance
from novelties.pricing.price_calculator import calculate_price
import itertools

class Simulator(object):
    def __init__(self, start_time, timestep):
        self.reset(start_time, timestep)
        sim_logger.setup_logging(self)
        self.logger = getLogger(__name__)
        self.demand_generator = DemandGenerator()
        self.routing_engine = RoutingEngine.create_engine()
        self.route_cache = {}
        self.current_dummyV = 0
        self.current_dqnV = 0

    def reset(self, start_time=None, timestep=None):
        if start_time is not None:
            self.__t = start_time
        if timestep is not None:
            self.__dt = timestep
        VehicleRepository.init()
        CustomerRepository.init()

    def populate_vehicle(self, vehicle_id, location):
        type = agent_codes.dqn_agent
        self.current_dqnV += 1
        # r = randrange(2)
        # if r == 0 and self.current_dummyV < FLAGS.dummy_vehicles:
        #     type = agent_codes.dummy_agent
        #     self.current_dummyV += 1
        #
        # # If r = 1 or num of dummy agent satisfied
        # elif self.current_dqnV < FLAGS.dqn_vehicles:
        #     type = agent_codes.dqn_agent
        #     self.current_dqnV += 1
        #
        # else:
        #     type = agent_codes.dummy_agent
        #     self.current_dummyV += 1

        VehicleRepository.populate(vehicle_id, location, type)

    def step(self):
        for customer in CustomerRepository.get_all():
            customer.step(self.__dt)
            if customer.is_arrived() or customer.is_disappeared():
                CustomerRepository.delete(customer.get_id())

        for vehicle in VehicleRepository.get_all():
            vehicle.step(self.__dt)
            # vehicle.print_vehicle()
            if vehicle.exit_market():
                score = ','.join(map(str, [self.get_current_time(), vehicle.get_id(), vehicle.get_total_dist(),
                                           vehicle.compute_profit()] + vehicle.get_score()))
                if vehicle.state.agent_type == agent_codes.dqn_agent:
                    self.current_dqnV -= 1
                else:
                    self.current_dummyV -= 1
                sim_logger.log_score(score)
                VehicleRepository.delete(vehicle.get_id())

        self.__populate_new_customers()
        self.__update_time()
        if self.__t % 3600 == 0:
            # print("Elapsed : {}".format(get_local_datetime(self.__t)))
            self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))


    def match_vehicles(self, commands, dqn_agent, dummy_agent):
        # print("M: ", commands)
        vehicle_list = []
        rejected_requests = []
        accepted_commands = []
        num_accepted = 0
        # reject_count = 0
        vehicle_accepted_cust = defaultdict(list)
        # od_accepted_pairs = []
        # Comamnd is a dictionary created in dummy_agent
        # print("########################################################")
        for command in commands:
            rejected_flag = 0
            # print(command["vehicle_id"], command["customer_id"])
            vehicle = VehicleRepository.get(command["vehicle_id"])
            vid = command["vehicle_id"]
            # print("V_Loc: ", vehicle.get_location())
            # vehicle.state.status = status_codes.V_ASSIGNED
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue
            # print("Vid: ", vid, "Plan: ", vehicle.current_plan)
            # print(vehicle.ordered_pickups_dropoffs_ids, vehicle.pickup_flags)
            # vehicle_cust_price_time = dict()

            # if len(vehicle.current_plan) != len(vehicle.current_plan_routes) != len( \
            #         vehicle.ordered_pickups_dropoffs_ids) != len(vehicle.pickup_flags):
            #     print("ERROR!")

            if FLAGS.enable_pooling:
                # print("vid: ", vehicle.get_id(), "Accepted: ", len(vehicle_accepted_cust[vid]))
                vehicle.state.tmp_capacity = vehicle.state.current_capacity

                # print("vid: ", vehicle.get_id(), len(vehicle_accepted_cust[vid]), "Final Plan: ", vehicle.current_plan,
                #                               vehicle.ordered_pickups_dropoffs_ids, len(vehicle.current_plan_routes), vehicle.pickup_flags)
                if len(vehicle.current_plan) == 0:
                    # print(vid, "EMPTYYYYYY!")
                    continue

                else:
                    route, triptime = vehicle.current_plan_routes.pop(0)
                    vehicle.nxt_stop = vehicle.current_plan.pop(0)
                    if len(route) == 0:
                        # print("B: ", triptime, len(vehicle.current_plan), len(vehicle.ordered_pickups_dropoffs_ids),
                        #       len(vehicle.current_plan_routes))
                        r2 = self.routing_engine.route_time([(vehicle.get_location(), vehicle.nxt_stop)])
                        route, triptime = r2[0]
                        # print("Updated: ", triptime, len(route))
                    # print("Loc: ", vehicle.get_location(), "Nxt: ", vehicle.nxt_stop)
                    cust_id = vehicle.ordered_pickups_dropoffs_ids[0]
                    if triptime == 0.0:
                        # vehicle.current_plan.pop(0)
                        pick_drop = vehicle.pickup_flags.pop(0)
                        cust_id = vehicle.ordered_pickups_dropoffs_ids.pop(0)

                        vehicle.state.assigned_customer_id = cust_id
                        if pick_drop == 1:
                            vehicle.state.lat, vehicle.state.lon = CustomerRepository.get(cust_id).get_origin()
                            vehicle.pickup(CustomerRepository.get(cust_id))
                        else:
                            vehicle.state.lat, vehicle.state.lon = CustomerRepository.get(cust_id).get_destination()
                            vehicle.dropoff(CustomerRepository.get(cust_id))
                    else:
                        vehicle.head_for_customer(triptime, cust_id, route)
                        # vehicle.nxt_stop = vehicle.current_plan[0]
                        # vehicle.change_to_assigned()

        # return accepted_commands

    def dispatch_vehicles(self, commands):
        # print("D: ", commands)
        od_pairs = []
        vehicles = []
        # Comamnd is a dictionary created in dummy_agent
        for command in commands:
            vehicle = VehicleRepository.get(command["vehicle_id"])
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue

            if "offduty" in command:
                off_duration = self.sample_off_duration()   # Rand time to rest
                vehicle.take_rest(off_duration)
            elif "cache_key" in command:
                l, a = command["cache_key"]
                route, triptime = self.routing_engine.get_route_cache(l, a)
                vehicle.cruise(route, triptime)
            else:
                vehicles.append(vehicle)
                od_pairs.append((vehicle.get_location(), command["destination"]))

        routes = self.routing_engine.route(od_pairs)

        for vehicle, (route, triptime) in zip(vehicles, routes):
            if triptime == 0:
                continue
            vehicle.cruise(route, triptime)

    def __update_time(self):
        self.__t += self.__dt

    def __populate_new_customers(self):
        new_customers = self.demand_generator.generate(self.__t, self.__dt)
        CustomerRepository.update_customers(new_customers)

    def sample_off_duration(self):
        return np.random.randint(OFF_DURATION / 2, OFF_DURATION * 3 / 2)

    def sample_pickup_duration(self):
        return np.random.exponential(PICKUP_DURATION)

    def get_current_time(self):
        t = self.__t
        return t

    def get_new_requests(self):
        return CustomerRepository.get_new_requests()

    def get_vehicles_state(self):
        return VehicleRepository.get_states()

    def get_vehicles(self):
        return VehicleRepository.get_all()

    def get_customers(self):
        return CustomerRepository.get_all()
        # return [VehicleRepository.get(id) for id in v_ids]

    # def log_score(self):
    #     for vehicle in VehicleRepository.get_all():
    #         score = ','.join(map(str, [self.get_current_time(), vehicle.get_id()] + vehicle.get_score()))
    #         sim_logger.log_score(score)
