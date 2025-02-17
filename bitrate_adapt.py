from bitrate_utils import calculate_tx_throughput, update_queue_bitrate
import time
import os
from dotenv import load_dotenv

load_dotenv()
THRESHOLD = os.getenv('THRESHOLD')
INTERVAL = os.getenv('INTERVAL')
NEW_BITRATE = os.getenv('NEW_BITRATE')
DEFAULT_BITRATE = os.getenv('DEFAULT_BITRATE')

if __name__ == "__main__":
    while True:
        tx_throughput = calculate_tx_throughput(interval=INTERVAL)

        if tx_throughput is not None:
            print(f"Current Transmitted Throughput: {tx_throughput:.2f} Mbps")
            if tx_throughput > THRESHOLD:
                update_queue_bitrate(NEW_BITRATE)
            else:
                update_queue_bitrate(DEFAULT_BITRATE)

        time.sleep(INTERVAL)
