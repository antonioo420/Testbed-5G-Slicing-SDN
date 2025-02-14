import requests
import time

# ODL API
ODL_IP = "158.49.247.131"
NODE_ID = "openflow:1" # BUSCAR
PORT_ID = "openflow:1:2" # BUSCAR
ODL_URL = "http://{ODL_IP}:8181/restconf/operational/opendaylight-inventory:nodes/node/{NODE_ID}/node-connector/{PORT_ID}" # CONFIRMAR
ODL_QUEUE_URL = "http://{ODL_IP}:8181/restconf/config/" #BUSCAR RUTA
AUTH = ("admin", "admin")

# UPDATE PARAMETERS
INTERVAL = 8
THRESHOLD = 1  # Mbps 
DEFAULT_BITRATE = 1200000  # 1.2 Mbps
NEW_BITRATE = 2000000  # 2 Mbps

# mientras no se agrega otra cola
QUEUE_ID = 10

def get_tx_bytes():
    response = requests.get(ODL_URL, auth=AUTH)
    if response.status_code == 200:
        data = response.json()
        try:
            tx_bytes = int(data["node-connector"][0]["opendaylight-port-statistics:flow-capable-node-connector-statistics"]["bytes"]["transmitted"])
            return tx_bytes
        except KeyError:
            print("Error: 'transmitted' field not found in API response.")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def calculate_tx_throughput(interval=2):
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
        "queue": {
            "id": "10",
            "other-config": {
                "max-rate": str(new_rate)
            }
        # REVISAR FORMA DEL JSON EN NOTAS
        }
    }

    response = requests.put(ODL_QUEUE_URL, json=queue_data, auth=AUTH)
    if response.status_code in [200, 204]:
        print(f"Successfully updated queue 10 max-rate to {new_rate} bps")
    else:
        print(f"Error: {response.status_code} - {response.text}")