from machine import ADC,Pin
import time
import sys
# import utime

adc = ADC(Pin(26))
voltage = 0
maxvol = 0
raw = 0
print("start")
while True:
        cmd = sys.stdin.readline().strip()
        if cmd == "READ":
            maxvol = 0
            i = 0
            while (i < 5):
                i = i + 1
                raw = adc.read_u16()
                voltage = raw * 3.3 / 65535
                if (voltage > maxvol):
                    maxvol = voltage

            print(maxvol)

#     utime.sleep(0.01)
#     print("ADC Raw:", raw,"Voltage:", voltage,"V",maxvol,"Vmax")

