try:
    from gpiozero import PWMOutputDevice, DigitalOutputDevice
except Exception:
    PWMOutputDevice = None
    DigitalOutputDevice = None

import time

b_multi = 2.0

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
        if PWMOutputDevice is None or DigitalOutputDevice is None:
            raise RuntimeError("gpiozero not available")

        self.in1 = DigitalOutputDevice(in1)
        self.in2 = DigitalOutputDevice(in2)
        self.in3 = DigitalOutputDevice(in3)
        self.in4 = DigitalOutputDevice(in4)

        self.ena = PWMOutputDevice(ena, initial_value=0.0)
        self.enb = PWMOutputDevice(enb, initial_value=0.0)

    def stop(self):
        self.ena.value = 0.0
        self.enb.value = 0.0 * b_multi
        self.in1.off()
        self.in2.off()
        self.in3.off()
        self.in4.off()

    def forward(self, speed=0.6):
        self.in1.on()
        self.in2.off()
        self.in3.on()
        self.in4.off()
        self.ena.value = speed
        self.enb.value = speed * b_multi

    def backward(self, speed=0.6):
        self.in1.off()
        self.in2.on()
        self.in3.off()
        self.in4.on()
        self.ena.value = speed
        self.enb.value = speed * b_multi

    def turn_left(self, speed=0.6):
        # Skid steer: left backward, right forward
        self.in1.on()
        self.in2.off()
        self.in3.off()
        self.in4.on()
        self.ena.value = speed
        self.enb.value = speed * b_multi

    def turn_right(self, speed=0.6):
        # Skid steer: left forward, right backward
        self.in1.off()
        self.in2.on()
        self.in3.on()
        self.in4.off()
        self.ena.value = speed
        self.enb.value = speed * b_multi

    def cleanup(self):
        self.stop()

if __name__ == "__main__":
    print("Motor controller test")
    motors = MotorController()
    motors.forward()
    time.sleep(3)
    motors.stop()