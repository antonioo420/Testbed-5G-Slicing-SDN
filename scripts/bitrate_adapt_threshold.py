from bitrate_utils import calculate_throughput, update_queue_bitrate
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
THRESHOLD = 25  # 25 Mbps 
DEFAULT_BR_1 = 30000000  # 30 Mbps
NEW_BR_1 = 60000000  # 60 Mbps
TOTAL_BR = 80000000  # 80 Mbps for current scenario

previous_state = False
update_queue_bitrate(new_rate=DEFAULT_BR_1, urlflow=URL_QUEUE_1, queue=QUEUE_1)
update_queue_bitrate(new_rate=TOTAL_BR - DEFAULT_BR_1, urlflow=URL_QUEUE_2, queue=QUEUE_2)

while True:
    throughput = calculate_throughput(interval=INTERVAL, urlstats=URL_STATS_1)
    
    if throughput is not None:
        print(f"Current throughput for Slice 1 is: {throughput:.2f} Mbps")    
        current_state = throughput > THRESHOLD

        if current_state != previous_state:
            if current_state:
                update_queue_bitrate(new_rate=NEW_BR_1, urlflow=URL_QUEUE_1, queue=QUEUE_1)
                update_queue_bitrate(new_rate=TOTAL_BR-NEW_BR_1, urlflow=URL_QUEUE_2, queue=QUEUE_2)
            else:
                update_queue_bitrate(new_rate=DEFAULT_BR_1, urlflow=URL_QUEUE_1, queue=QUEUE_1)
                update_queue_bitrate(new_rate=TOTAL_BR-DEFAULT_BR_1, urlflow=URL_QUEUE_2, queue=QUEUE_2)
            
            previous_state = current_state