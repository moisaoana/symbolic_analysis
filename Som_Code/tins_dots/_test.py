# read event codes
from BinaryFileReader import BinaryFileReader

binaryFileReader = BinaryFileReader()

event_codes = binaryFileReader.read_event_codes('Dots_30_001-Event-Codes.bin')

print(event_codes)
print(event_codes.size)
