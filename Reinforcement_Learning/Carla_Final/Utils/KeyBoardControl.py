import random

import carla
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q


class KeyboardControl(object):
    def __init__(self) -> None:
        super().__init__()

    def parse_events(self, world : carla.World, plan, client: carla.Client):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
            if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d: #Activate debug
                        return 0
                    if event.key == pygame.K_1:
                        return 1
                    if event.key == pygame.K_2:
                        return 2
                    if event.key == pygame.K_3:
                        return 3
                    if event.key == pygame.K_4:
                        return 4
                    if event.key == pygame.K_5:
                        return 5
                    if event.key == pygame.K_6:
                        return 6
                    if event.key == pygame.K_7:
                        return 7
                    if event.key == pygame.K_r:
                        return 10
                    if event.key == pygame.K_s:
                        return 11
                    if event.key == pygame.K_o:
                        return 12
                    if event.key == pygame.K_k:
                        return 13
                    if event.key == pygame.K_l:
                        return 14


    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
