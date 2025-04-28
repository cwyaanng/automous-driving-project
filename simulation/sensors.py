import math
import numpy as np
import weakref
import pygame
from simulation.connection import carla
from simulation.settings import RGB_CAMERA, SSC_CAMERA


# ---------------------------------------------------------------------|
# ------------------------------- CAMERA |
# ---------------------------------------------------------------------|

class CameraSensor():

    def __init__(self, vehicle):
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'160')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target/255.0)


# ---------------------------------------------------------------------|
# ------------------------------- ENV CAMERA |
# ---------------------------------------------------------------------|

class CameraSensorEnv:
    def __init__(self, vehicle):
        self.sensor_name = 'sensor.camera.rgb'
        self.parent = vehicle
        self.front_camera = []  # 앞 카메라 센서 이미지 저장 
        self.rear_camera = [] # 뒤 카메라 센서 이미지 저장 

        world = self.parent.get_world()
        self.front_sensor = self._set_camera_sensor(world, position='front')
        self.rear_sensor = self._set_camera_sensor(world, position='rear')
        weak_self = weakref.ref(self)
        self.front_sensor.listen(lambda image: CameraSensorEnv._save_image(weak_self, image, 'front'))
        self.rear_sensor.listen(lambda image: CameraSensorEnv._save_image(weak_self, image, 'rear'))

    def _set_camera_sensor(self, world,position='front'):
        blueprint = world.get_blueprint_library().find(self.sensor_name)
        blueprint.set_attribute('image_size_x', '1280')
        blueprint.set_attribute('image_size_y', '720')
        blueprint.set_attribute('fov', '90')
        blueprint.set_attribute('sensor_tick', '0.05')

        if position == 'front':
            transform = carla.Transform(
            carla.Location(x=6.0, z=4.0),       # 차량 앞쪽 위치
            carla.Rotation(pitch=-15.0, yaw=180.0)  # 뒤로 돌려서 차량 바라봄
            )
        elif position == 'rear':
            transform = carla.Transform(
                carla.Location(x=-6.0, z=4.0),      # 차량 뒤쪽 위치
                carla.Rotation(pitch=-15.0, yaw=0.0)    # 그대로 차량 바라봄
            )


        sensor = world.spawn_actor(blueprint, transform, attach_to=self.parent)
        return sensor


    @staticmethod
    def _save_image(weak_self, image, position):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb_array = array[:, :, :3]  # RGB만 추출

        if position == 'front':
            self.front_camera.append(rgb_array)
        elif position == 'rear':
            self.rear_camera.append(rgb_array)



# ---------------------------------------------------------------------|
# ------------------------------- COLLISION SENSOR|
# ---------------------------------------------------------------------|

# It's an important as it helps us to tract collisions
# It also helps with resetting the vehicle after detecting any collisions
class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)

