from bitrate_utils import calculate_tx_throughput, update_queue_bitrate
import os
from dotenv import load_dotenv

load_dotenv()
THRESHOLD = os.getenv('THRESHOLD')
NEW_BITRATE = os.getenv('NEW_BITRATE')
DEFAULT_BITRATE = os.getenv('DEFAULT_BITRATE')

tx_throughput = calculate_tx_throughput()

if tx_throughput is not None:
    print(f"Current Transmitted Throughput: {tx_throughput:.2f} Mbps")
    if tx_throughput > THRESHOLD:
        update_queue_bitrate(NEW_BITRATE)
    else:
        update_queue_bitrate(DEFAULT_BITRATE)
