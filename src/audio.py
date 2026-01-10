import os
import pygame

class AudioManager:
    def __init__(self):
        pygame.mixer.init()
        
        audio_dir = os.path.join('voice', 'commands')
        self.audio_files = {
            'STOP': os.path.join(audio_dir, 'stop.mp3'),
            'OBSTACLE': os.path.join(audio_dir, 'obstacle_detected.mp3'),
            'GO': os.path.join(audio_dir, 'go.mp3'),
            'TURN_LEFT': os.path.join(audio_dir, 'turn_left.mp3'),
            'TURN_RIGHT': os.path.join(audio_dir, 'turn_right.mp3'),
        }
        
        self.enabled = all(os.path.exists(f) for f in self.audio_files.values())
        
        if self.enabled:
            print("✓ Audio enabled")
        else:
            print("⚠ Some audio files missing - running without audio")
    
    def play(self, command):
        if not self.enabled or command not in self.audio_files:
            return
        
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(self.audio_files[command])
            pygame.mixer.music.play()
