from bitrate_utils import calculate_throughput, update_queue_bitrate
import time
import os
from dotenv import load_dotenv

# Environment parameters
load_dotenv()
URL_STATS_1 = os.getenv('ODL_STATS1_URL')
URL_QUEUE_1 = os.getenv('ODL_QUEUE1_URL')
URL_QUEUE_2 = os.getenv('ODL_QUEUE2_URL')
QUEUE_1 = os.getenv('QUEUE1_ID')
QUEUE_2 = os.getenv('QUEUE2_ID')

# Update parameters
INTERVAL = 5
THRESHOLD = 1  # 1 Mbps 
DEFAULT_BR_1 = 1200000  # 1.2 Mbps
NEW_BR_1 = 2000000  # 2 Mbps
TOTAL_BR = 90000000  # 90 Mbps for current scenario

previous_state = None

while True:
    throughput = calculate_throughput(interval=INTERVAL, urlstats=URL_STATS_1)
    
    if throughput is not None:
        print(f"Current throughput for Slice 1 is: {throughput:.2f} Mbps")    
        current_state = throughput > THRESHOLD

        if current_state != previous_state:
            if current_state:  # Transitioned to above threshold
                update_queue_bitrate(new_rate=NEW_BR_1, urlflow=URL_QUEUE_1, queue=QUEUE_1)
                update_queue_bitrate(new_rate=TOTAL_BR - NEW_BR_1, urlflow=URL_QUEUE_2, queue=QUEUE_2)
            else:  # Transitioned to below threshold
                update_queue_bitrate(new_rate=NEW_BR_1, urlflow=URL_QUEUE_1, queue=QUEUE_1)
                update_queue_bitrate(new_rate=TOTAL_BR - NEW_BR_1, urlflow=URL_QUEUE_2, queue=QUEUE_2)
            
            previous_state = current_state  # Update the state
    
    time.sleep(INTERVAL)