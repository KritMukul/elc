import time
import sys
import logging
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eeg_acquire")

try:
    import board
    import busio
    from adafruit_ads1x15.analog_in import AnalogIn
    import adafruit_ads1x15.ads1115 as ADS_mod
    from adafruit_ads1x15.ads1115 import ADS1115 as ADS1115_class
except ImportError as e:
    log.error("Adafruit libraries missing. Run: pip install adafruit-circuitpython-ads1x15")
    sys.exit(1)

def bind_channels(ads):
    chans = []
    try:
        for i in range(4):
            chans.append(AnalogIn(ads, i))
        return chans
    except Exception as e:
        log.error("Failed to bind channels: %s", e)
        return []

def main():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS1115_class(i2c)
        # To get higher sampling rates, you may need to adjust data_rate depending on ADS library version
        # ads.data_rate = 860 
    except Exception as e:
        log.error("Hardware init failed. Are the pins connected correctly? Error: %s", e)
        sys.exit(1)

    chans = bind_channels(ads)
    if not chans:
        sys.exit(1)

    # Prepare CSV File for Logging
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_eeg_data_{timestamp_str}.csv"
    
    log.info("Starting acquisition. Press Ctrl+C to stop.")
    log.info(f"Data will be saved to: {filename}")

    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Header
            writer.writerow(["Timestamp", "Ch0_V", "Ch1_V", "Ch2_V", "Ch3_V"])
            
            start_time = time.time()
            sample_count = 0
            
            while True:
                current_time = time.time() - start_time
                readings = [current_time]
                
                for ch in chans:
                    try:
                        readings.append(ch.voltage)
                    except Exception:
                        readings.append(float('nan'))
                
                writer.writerow(readings)
                sample_count += 1
                
                # Print progress every 100 samples
                if sample_count % 100 == 0:
                    print(f"Recorded {sample_count} samples...", end='\r')
                    
                # No sleep here to capture as fast as possible for EEG
                # Python on Pi with ADS1115 typically maxes out around 100-250Hz in a simple loop

    except KeyboardInterrupt:
        log.info("\nAcquisition stopped by user.")
        log.info(f"Total samples recorded: {sample_count}")
        log.info(f"File saved: {filename}")

if __name__ == "__main__":
    main()
