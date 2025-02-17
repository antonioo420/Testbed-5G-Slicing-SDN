import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
URL_STATS_1 = os.getenv('ODL_STATS1_URL')
URL_QUEUE_1 = os.getenv('ODL_QUEUE1_URL')
QUEUE_1 = os.getenv('QUEUE1_ID')
AUTH = os.getenv('AUTH')


def get_tx_bytes():
    response = requests.get(URL_STATS_1, auth=AUTH)
    if response.status_code == 200:
        data = response.json()
        try:
            tx_bytes = data["opendaylight-flow-statistics:flow-statistics"]
            tx_trans = int(tx_bytes["byte-count"])
            #print(tx_trans)
            return tx_trans
        except KeyError:
            print("Error: 'transmitted' field not found in API response.")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def calculate_tx_throughput(interval):
    tx_bytes_before = get_tx_bytes()
    if tx_bytes_before is None:
        return None

    time.sleep(interval)

    tx_bytes_after = get_tx_bytes()
    if tx_bytes_after is None:
        return None

    tx_throughput = (tx_bytes_after - tx_bytes_before) * 8 / 1e6 / interval  # Mbps
    return tx_throughput


def update_queue_bitrate(new_rate):
    queue_data = {
        "ovsdb:queues": [
            {
                "queue-id": f"queue://{QUEUE_1}",
                "queue-uuid": f"{QUEUE_1}",
                "queues-other-config": [
                    {
                        "queue-other-config-key": "max-rate",
                        "queue-other-config-value": str(new_rate)
                    }
                ]
            }
        ]
    }

    response = requests.put(URL_QUEUE_1, json=queue_data, auth=AUTH)
    if response.status_code in [200, 204]:
        print(f"Successfully updated queue 10 max-rate to {new_rate} bps")
    else:
        print(f"Error: {response.status_code} - {response.text}")
