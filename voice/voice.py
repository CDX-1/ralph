import os
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
import pygame

load_dotenv()

class VoiceCommands:
    def __init__(self):
        self.api_key = os.getenv('ELEVENLABS_KEY')
        if not self.api_key:
            raise ValueError("ELEVENLABS_KEY not found in .env file")
        
        self.client = ElevenLabs(api_key=self.api_key)
        
        pygame.mixer.init()
        
        self.cache_dir = "voice_commands"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.commands = {}
    
    def generate_command(self, text, voice="Rachel", save_name=None):
        if save_name is None:
            save_name = text.lower().replace(" ", "_")
        
        filepath = os.path.join(self.cache_dir, f"{save_name}.mp3")
        
        if os.path.exists(filepath):
            print(f"Using cached: {filepath}")
            return filepath
        
        print(f"Generating: '{text}'...")
        
        try:
            audio = self.client.generate(
                text=text,
                voice=voice,
                model="eleven_multilingual_v2"
            )
            
            with open(filepath, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
            
            print(f"Saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error generating voice: {e}")
            return None
    
    def play_command(self, command_name):
        if command_name not in self.commands:
            print(f"Command '{command_name}' not found. Generate it first.")
            return
        
        filepath = self.commands[command_name]
        
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def generate_safety_commands(self):
        commands = {
            "stop": "Stop",
            "safe": "Safe to proceed",
            "danger": "Danger ahead",
            "caution": "Caution",
            "clear": "Path is clear",
            "obstacle_left": "Obstacle on the left",
            "obstacle_right": "Obstacle on the right",
            "obstacle_center": "Obstacle ahead",
            "turn_left": "Turn left",
            "turn_right": "Turn right",
            "slow_down": "Slow down",
            "all_clear": "All clear"
        }
        
        for name, text in commands.items():
            filepath = self.generate_command(text, save_name=name)
            if filepath:
                self.commands[name] = filepath
        
        print(f"\n✓ Generated {len(self.commands)} voice commands")
        return self.commands


def main():
    print("=" * 50)
    print("ElevenLabs Voice Command Generator")
    print("=" * 50)
    
    try:
        vc = VoiceCommands()
        
        print("\nGenerating safety commands...")
        vc.generate_safety_commands()
        
        print("\n" + "=" * 50)
        print("Testing voice commands...")
        print("=" * 50)
        
        test_commands = ["stop", "safe", "danger", "clear"]
        
        for cmd in test_commands:
            print(f"\nPlaying: {cmd}")
            vc.play_command(cmd)
            pygame.time.wait(500)
        
        print("\n✓ All tests completed!")
        print(f"\nVoice files saved in: {vc.cache_dir}/")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("1. ELEVENLABS_KEY is set in .env file")
        print("2. You have installed: pip install elevenlabs python-dotenv pygame")


if __name__ == "__main__":
    main()
