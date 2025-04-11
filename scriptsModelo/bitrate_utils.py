import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
URL_STATS_1 = os.getenv('ODL_STATS1_URL')
AUTH = os.getenv('AUTH')


def get_bytes(urlstats):
    response = requests.get(urlstats, auth=("admin","admin"))
    if response.status_code == 200:
        data = response.json()
        try:
            tx_bytes = data["opendaylight-flow-statistics:flow-statistics"]
            instant_throughput = int(tx_bytes["byte-count"])
            #print(instant_throughput)
            return instant_throughput
        except KeyError:
            print("Error: field not found in API response.")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def calculate_throughput(interval, urlstats):
    bytes_before = get_bytes(urlstats)
    if bytes_before is None:
        return None

    time.sleep(interval)

    bytes_after = get_bytes(urlstats)
    if bytes_after is None:
        return None

    throughput = (bytes_after - bytes_before) * 8 / 1e6 / interval  # Mbps
    return throughput


def update_queue_bitrate(new_rate,urlflow,queue):
    queue_data = {
        "ovsdb:queues": [
            {
                "queue-id": f"queue://{queue}",
                "queue-uuid": f"{queue}",
                "queues-other-config": [
                    {
                        "queue-other-config-key": "max-rate",
                        "queue-other-config-value": str(new_rate)
                    }
                ]
            }
        ]
    }

    response = requests.put(urlflow, json=queue_data, auth=("admin","admin"))
    if response.status_code in [200, 204]:
        print(f"Queue max-rate updated to {new_rate} bps")
    else:
        print(f"Error: {response.status_code} - {response.text}")
