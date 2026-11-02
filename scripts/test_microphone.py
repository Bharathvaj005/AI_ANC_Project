import sounddevice as sd

print("=" * 50)
print("AVAILABLE AUDIO DEVICES")
print("=" * 50)

print(sd.query_devices())

print("\n" + "=" * 50)
print("DEFAULT INPUT DEVICE")
print("=" * 50)

print(sd.query_devices(kind="input"))