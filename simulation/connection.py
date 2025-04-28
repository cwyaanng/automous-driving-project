import os
import sys
import glob

try:
    sys.path.append(os.path.abspath("/home/jini70899/ad/auto/carla/carla-0.9.8-py3.5-linux-x86_64.egg"))
except IndexError:
    print('Couldn\'t import Carla egg properly')

import carla
from simulation.settings import PORT, TIMEOUT, HOST

class ClientConnection:
    def __init__(self, town):
        self.client = None
        self.town = town

    def setup(self):
        try:

            # Connecting to the  Server
            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            return self.client, self.world

        except Exception as e:
            print(
                'Failed to make a connection with the server: {}'.format(e))
            print(self.client.get_available_maps())
            self.error()

    # An error method: prints out the details if the client failed to make a connection
    def error(self):

        print("\nClient version: {}".format(
            self.client.get_client_version()))
        print("Server version: {}\n".format(
            self.client.get_server_version()))

        if self.client.get_client_version() != self.client.get_server_version():
            print(
                "There is a Client and Server version mismatch! Please install or download the right versions.")
