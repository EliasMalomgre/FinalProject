import carla
import weakref
from carla import ColorConverter as cc
import numpy as np
import pygame

from Carla_Final.CONFIG import CONFIG


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor):
        self.camera_detect = None
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.image = None

    def set_sensor(self):
        bp_library = self._parent.get_world().get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=0, z=2))
        self.camera_detect = self._parent.get_world().spawn_actor(camera_bp, camera_transform, attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.camera_detect.listen(lambda image: self._parse_image_detect(weak_self, image))


    def render(self, display):
        if self.image is not None:
            self.surface = pygame.surfarray.make_surface(self.image)
            self.surface = pygame.transform.flip(self.surface, True, False)
            self.surface = pygame.transform.rotate(self.surface, 90)
            display.blit(self.surface, (0, 0))

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    @staticmethod
    def _parse_image_detect(weak_self, image):
        self = weak_self()
        if not self:
            return
        im_array = np.copy(np.frombuffer(image.raw_data, dtype=np.dtype("uint8")))
        im_array = np.reshape(im_array, (image.height, image.width, 4))
        im_array = im_array[:, :, :3][:, :, ::-1]

        self.image = im_array

