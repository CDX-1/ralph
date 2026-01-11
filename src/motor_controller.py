"""
Motor Controller for 2-Wheel Robot
Supports L298N motor driver or similar H-bridge drivers
"""

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    print("⚠ RPi.GPIO not available - Motor control in simulation mode")


class MotorController:
    """
    Controls 2 DC motors for a differential drive robot
    
    Typical L298N connections:
    - Left Motor: ENA (speed), IN1, IN2 (direction)
    - Right Motor: ENB (speed), IN3, IN4 (direction)
    """
    
    def __init__(self, 
                 left_forward_pin=17, 
                 left_backward_pin=27,
                 right_forward_pin=22,
                 right_backward_pin=23,
                 left_enable_pin=12,
                 right_enable_pin=13,
                 pwm_frequency=1000,
                 default_speed=70):
        """
        Initialize motor controller with GPIO pins
        
        Args:
            left_forward_pin: GPIO pin for left motor forward
            left_backward_pin: GPIO pin for left motor backward
            right_forward_pin: GPIO pin for right motor forward
            right_backward_pin: GPIO pin for right motor backward
            left_enable_pin: GPIO pin for left motor speed (PWM)
            right_enable_pin: GPIO pin for right motor speed (PWM)
            pwm_frequency: PWM frequency in Hz
            default_speed: Default speed (0-100%)
        """
        self.left_forward = left_forward_pin
        self.left_backward = left_backward_pin
        self.right_forward = right_forward_pin
        self.right_backward = right_backward_pin
        self.left_enable = left_enable_pin
        self.right_enable = right_enable_pin
        
        self.default_speed = default_speed
        self.current_action = "STOP"
        self.simulation_mode = not GPIO_AVAILABLE
        
        if GPIO_AVAILABLE:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup motor control pins
            GPIO.setup(self.left_forward, GPIO.OUT)
            GPIO.setup(self.left_backward, GPIO.OUT)
            GPIO.setup(self.right_forward, GPIO.OUT)
            GPIO.setup(self.right_backward, GPIO.OUT)
            GPIO.setup(self.left_enable, GPIO.OUT)
            GPIO.setup(self.right_enable, GPIO.OUT)
            
            # Setup PWM for speed control
            self.left_pwm = GPIO.PWM(self.left_enable, pwm_frequency)
            self.right_pwm = GPIO.PWM(self.right_enable, pwm_frequency)
            
            self.left_pwm.start(0)
            self.right_pwm.start(0)
            
            print("✓ Motor controller initialized (GPIO mode)")
        else:
            print("✓ Motor controller initialized (Simulation mode)")
            self.left_pwm = None
            self.right_pwm = None
    
    def _set_motor_pins(self, left_fwd, left_bwd, right_fwd, right_bwd):
        """Set individual motor pin states"""
        if GPIO_AVAILABLE:
            GPIO.output(self.left_forward, left_fwd)
            GPIO.output(self.left_backward, left_bwd)
            GPIO.output(self.right_forward, right_fwd)
            GPIO.output(self.right_backward, right_bwd)
    
    def _set_speed(self, left_speed, right_speed):
        """Set PWM duty cycle for motor speeds (0-100)"""
        if GPIO_AVAILABLE:
            self.left_pwm.ChangeDutyCycle(max(0, min(100, left_speed)))
            self.right_pwm.ChangeDutyCycle(max(0, min(100, right_speed)))
    
    def forward(self, speed=None):
        """Move forward at specified speed"""
        speed = speed or self.default_speed
        self._set_motor_pins(True, False, True, False)
        self._set_speed(speed, speed)
        self.current_action = "FORWARD"
        if self.simulation_mode:
            print(f"🤖 Motor: FORWARD at {speed}%")
    
    def backward(self, speed=None):
        """Move backward at specified speed"""
        speed = speed or self.default_speed
        self._set_motor_pins(False, True, False, True)
        self._set_speed(speed, speed)
        self.current_action = "BACKWARD"
        if self.simulation_mode:
            print(f"🤖 Motor: BACKWARD at {speed}%")
    
    def turn_left(self, speed=None):
        """Turn left (left motor backward, right motor forward)"""
        speed = speed or self.default_speed
        self._set_motor_pins(False, True, True, False)
        self._set_speed(speed, speed)
        self.current_action = "TURN_LEFT"
        if self.simulation_mode:
            print(f"🤖 Motor: TURN LEFT at {speed}%")
    
    def turn_right(self, speed=None):
        """Turn right (left motor forward, right motor backward)"""
        speed = speed or self.default_speed
        self._set_motor_pins(True, False, False, True)
        self._set_speed(speed, speed)
        self.current_action = "TURN_RIGHT"
        if self.simulation_mode:
            print(f"🤖 Motor: TURN RIGHT at {speed}%")
    
    def pivot_left(self, speed=None):
        """Pivot left (both motors backward on left, forward on right - sharper turn)"""
        speed = speed or self.default_speed
        self._set_motor_pins(False, True, True, False)
        self._set_speed(speed * 0.5, speed)
        self.current_action = "PIVOT_LEFT"
        if self.simulation_mode:
            print(f"🤖 Motor: PIVOT LEFT at {speed}%")
    
    def pivot_right(self, speed=None):
        """Pivot right (both motors forward on left, backward on right - sharper turn)"""
        speed = speed or self.default_speed
        self._set_motor_pins(True, False, False, True)
        self._set_speed(speed, speed * 0.5)
        self.current_action = "PIVOT_RIGHT"
        if self.simulation_mode:
            print(f"🤖 Motor: PIVOT RIGHT at {speed}%")
    
    def stop(self):
        """Stop all motors"""
        self._set_motor_pins(False, False, False, False)
        self._set_speed(0, 0)
        self.current_action = "STOP"
        if self.simulation_mode:
            print("🤖 Motor: STOP")
    
    def gradual_stop(self, steps=5, delay=0.05):
        """Gradually reduce speed to stop (smoother)"""
        import time
        current_speed = self.default_speed
        for i in range(steps):
            current_speed = current_speed * (steps - i - 1) / steps
            self._set_speed(current_speed, current_speed)
            time.sleep(delay)
        self.stop()
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop()
        if GPIO_AVAILABLE:
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()
            print("✓ Motor controller cleanup complete")
        else:
            print("✓ Motor controller cleanup complete (Simulation mode)")
    
    def get_status(self):
        """Get current motor status"""
        return {
            "action": self.current_action,
            "simulation_mode": self.simulation_mode,
            "default_speed": self.default_speed
        }


if __name__ == "__main__":
    """Test motor controller"""
    import time
    
    print("\n=== Motor Controller Test ===\n")
    
    # Initialize with default pins
    motors = MotorController(default_speed=60)
    
    try:
        print("Testing each direction for 2 seconds...\n")
        
        motors.forward()
        time.sleep(2)
        
        motors.backward()
        time.sleep(2)
        
        motors.turn_left()
        time.sleep(2)
        
        motors.turn_right()
        time.sleep(2)
        
        motors.stop()
        print("\nTest complete!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        motors.cleanup()
