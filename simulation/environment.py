import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *
import os
import traceback
class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:

        print("[INIT] CarlaEnvironment initialized.")
        self.image_save_dir = None
        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = False  # 시각화 꺼버림

        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()



    def reset(self):
        print("\n[RESET] Called =================================")
        try:
            self.frame_count = 0
            print("[RESET] Calling remove_sensors()")
            self.remove_sensors()
            print(f"[RESET] Sensors after removal: {[x.id for x in self.sensor_list]}")
            print(f"[RESET] Actors after removal: {[x.id for x in self.actor_list]}")

            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                print("[RESET] Destroying previous actors/sensors...")
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
                print("[RESET] Cleared sensor and actor lists")

            print("[RESET] Getting vehicle blueprint")
            vehicle_bp = self.get_vehicle(CAR_NAME)

            print(f"[RESET] Spawning vehicle in town: {self.town}")
            if self.town == "Town07":
                transform = self.map.get_spawn_points()[38]
                self.total_distance = 750
            elif self.town == "Town02":
                transform = self.map.get_spawn_points()[1]
                self.total_distance = 780
            else:
                transform = random.choice(self.map.get_spawn_points())
                self.total_distance = 250

            print(f"[RESET] Trying to spawn vehicle at transform: {transform}")
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if not self.vehicle:
                raise RuntimeError("[RESET ERROR] Failed to spawn vehicle.")
            print(f"[RESET] Spawned vehicle: {self.vehicle.id}")
            self.actor_list.append(self.vehicle)

            # Camera Sensor
            print("[RESET] Initializing CameraSensor")
            self.camera_obj = CameraSensor(self.vehicle)
            self.sensor_list.append(self.camera_obj.sensor)
            print(f"[RESET] Camera sensor ID: {self.camera_obj.sensor.id}")
            
            print("[RESET] Waiting for camera images...")
            wait_counter = 0
            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.001)
                wait_counter += 1
                if wait_counter > 10000:
                    raise RuntimeError("[RESET ERROR] Camera image not received after waiting.")

            self.image_obs = self.camera_obj.front_camera.pop(-1)
            print("[RESET] Received initial camera image.")

            # Third person view of our vehicle in the Simulated env
            # 기존 코드 수정
            if self.display_on == False:
                print("[RESET] Initializing Env CameraSensor")
                self.env_camera_obj = CameraSensorEnv(self.vehicle)

                # 두 개 센서 모두 리스트에 추가!
                self.sensor_list.append(self.env_camera_obj.front_sensor)
                self.sensor_list.append(self.env_camera_obj.rear_sensor)

                print(f"[RESET] Env front camera sensor ID: {self.env_camera_obj.front_sensor.id}")
                print(f"[RESET] Env rear camera sensor ID: {self.env_camera_obj.rear_sensor.id}")


            # Collision sensor
            print("[RESET] Initializing CollisionSensor")
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)
            print(f"[RESET] Collision sensor ID: {self.collision_obj.sensor.id}")

            # Initialize other state vars
            print("[RESET] Initializing environment variables")
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22
            self.max_speed = 25.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3
            self.throttle = 0.0
            self.previous_steer = 0.0
            self.velocity = 0.0
            self.distance_from_center = 0.0
            self.angle = 0.0
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0

            if self.fresh_start:
                print("[RESET] Creating new route waypoints")
                self.current_waypoint_index = 0
                self.route_waypoints = []
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                print("[RESET] Using existing route waypoints and teleporting to checkpoint")
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.navigation_obs = np.array([
                self.throttle,
                self.velocity,
                self.previous_steer,
                self.distance_from_center,
                self.angle
            ])

            print("[RESET] Reset done successfully")
            time.sleep(0.5)
            self.collision_history.clear()
            self.episode_start_time = time.time()

            return [self.image_obs, self.navigation_obs]

        except Exception as e:
            print(f"[RESET ERROR] {e}")
            import traceback
            traceback.print_exc()

            self.remove_sensors()
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            except Exception as destroy_err:
                print(f"[RESET CLEANUP ERROR] Failed during batch destroy: {destroy_err}")

            self.sensor_list.clear()
            self.actor_list.clear()
            if self.display_on:
                print("[RESET] Quitting pygame")
                pygame.quit()



# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.fresh_start = False

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1))
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                self.throttle = 1.0
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data            

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()


            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

             # Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            # Rewards are given below!
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                done = True
                reward = -10
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                reward = -10
                done = True
            elif self.velocity > self.max_speed:
                reward = -10
                done = True

            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.continous_action_space:
                    if self.velocity < self.min_speed:
                        reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                    elif self.velocity > self.target_speed:               
                        reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    else:                                         
                        reward = 1.0 * centering_factor * angle_factor 
                else:
                    reward = 1.0 * centering_factor * angle_factor

            if self.timesteps >= 7500:
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            self.image_obs = self.camera_obj.front_camera.pop(-1)
            
            # ==========  이미지 저장 로직 ==============
            if self.save_image:
                from PIL import Image

                # Ensure the directory exists
                os.makedirs(self.image_save_dir, exist_ok=True)

                if self.env_camera_obj is not None and len(self.env_camera_obj.front_camera) > 0:
                    env_img = self.env_camera_obj.front_camera.pop(-1)
                    env_img_pil = Image.fromarray(env_img)
                    env_img_pil.save(os.path.join(self.image_save_dir, f"front_third_frame_{self.frame_count:05d}.png"))
                    
                    env_img = self.env_camera_obj.rear_camera.pop(-1)
                    env_img_pil = Image.fromarray(env_img)
                    env_img_pil.save(os.path.join(self.image_save_dir, f"rear_third_frame_{self.frame_count:05d}.png"))

                self.frame_count += 1


            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])
            
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                for sensor in self.sensor_list:
                    try:
                        sensor.destroy()
                    except Exception as e:
                        print(f"[STEP ERROR] Failed to destroy sensor {sensor.id}: {e}")
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    try:
                        actor.destroy()
                    except Exception as e:
                        print(f"[STEP ERROR] Failed to destroy actor {actor.id}: {e}")
            
            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        except Exception as e:
            print(f"[carla step error] {e}")
            traceback.print_exc()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()



# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        if self.camera_obj is not None and hasattr(self.camera_obj, 'sensor'):
            try:
                print(f"[CLEANUP] Trying to destroy sensor ")
                self.camera_obj.sensor.stop()
                self.camera_obj.sensor.destroy()
                print(f"[CLEANUP] Destroyed sensor")
            except Exception as e:
                print(f"[CLEANUP ERROR] Failed to destroy sensor: {e}")
                pass
            self.camera_obj = None

        if self.env_camera_obj is not None and hasattr(self.env_camera_obj, 'sensor'):
            try:
                self.env_camera_obj.sensor.stop()
                self.env_camera_obj.sensor.destroy()
            except:
                pass
            self.env_camera_obj = None

        if self.collision_obj is not None and hasattr(self.collision_obj, 'sensor'):
            try:
                self.collision_obj.sensor.stop()
                self.collision_obj.sensor.destroy()
            except:
                pass
            self.collision_obj = None

