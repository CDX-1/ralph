try:
    from gpiozero import Motor, PWMOutputDevice
except Exception:  # pragma: no cover - allows import on non-Pi hosts
    Motor = None
    PWMOutputDevice = None


class MotorController:
    def __init__(
        self,
        in1=17,
        in2=27,
        ena=22,
        in3=23,
        in4=24,
        enb=25,
    ):
        if Motor is None or PWMOutputDevice is None:
            raise RuntimeError("gpiozero not available")

        self.left_enable = PWMOutputDevice(ena, initial_value=0.0)
        self.right_enable = PWMOutputDevice(enb, initial_value=0.0)

        self.left = Motor(forward=in1, backward=in2, enable=self.left_enable)
        self.right = Motor(forward=in3, backward=in4, enable=self.right_enable)

    def stop(self):
        self.left.stop()
        self.right.stop()

    def forward(self, speed=0.6):
        self.left.forward(speed)
        self.right.forward(speed)

    def backward(self, speed=0.6):
        self.left.backward(speed)
        self.right.backward(speed)

    def turn_left(self, speed=0.6):
        # Skid steer: left backward, right forward
        self.left.backward(speed)
        self.right.forward(speed)

    def turn_right(self, speed=0.6):
        # Skid steer: left forward, right backward
        self.left.forward(speed)
        self.right.backward(speed)

    def cleanup(self):
        self.stop()
