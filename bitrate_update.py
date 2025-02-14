import bitrate_utils
import requests

tx_throughput = calculate_tx_throughput()

if tx_throughput is not None:
    print(f"Current Transmitted Throughput: {tx_throughput:.2f} Mbps")
    if tx_throughput > THRESHOLD:
        update_queue_bitrate(NEW_BITRATE)
    else:
        update_queue_bitrate(DEFAULT_BITRATE)
