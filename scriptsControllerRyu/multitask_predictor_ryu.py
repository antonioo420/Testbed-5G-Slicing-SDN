from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, ipv4, tcp, udp
from ryu.lib import dpid as dpid_lib

import requests
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, ReLU, SimpleRNN, GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict, deque
import pandas as pd
from sklearn.metrics import mean_squared_error   
import time
import json
from scapy.all import IP as scapy_IP, raw
import threading
import logging
import struct

LOOKBACK = 30
LAST_PRED = 10
#window = deque([0]* 30,maxlen=LOOKBACK)

class SimpleMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor, self).__init__(*args, **kwargs)
        self.model = load_model("/home/lab/traffic_predictor.h5", custom_objects={'mse': MeanSquaredError()})
        self.logger.info("Modelo cargado")
        self.datapath = None
        self.window = defaultdict(lambda: deque([0]* 30,maxlen=LOOKBACK))
        self.ues = set()
        self.dpid_str = {}
        self.ip_map = {}
        self.last_pred = defaultdict(lambda: deque(maxlen=LAST_PRED))
        logging.getLogger('ryu.controller.ofp_handler').setLevel(logging.WARNING)
        
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """ If switch connects creates flows and queues. If it disconnects deletes datapath """
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            #self.datapaths[datapath.id] = datapath
            self.datapath = datapath
            self.dpid_str = dpid_lib.dpid_to_str(datapath.id)
            self.add_default_flow(datapath)
            self.add_default_queue_flows(datapath)
            time.sleep(1)
            self.set_ovsdb_addr(self.dpid_str)
            time.sleep(1)
            self.create_queues(self.dpid_str, 1000000, 20000000)
            self.send_to_controller_from_port(datapath, 9)
            self.prediction_throughput_handler()
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def add_default_flow(self, datapath):
        """Adds flow "actions=normal" to the switch using OpenFlow """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()  # Match all packets
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]  # Actions normal

        # Crear e instalar la regla de flujo
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=100, match=match, instructions=inst)
        datapath.send_msg(mod)

        self.logger.info("Actions normal flow installed")

    def add_default_queue_flows(self, datapath):
        """ Adds queue flows for UPF1 and UPF2 downlink ports"""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # in_port=9 → queue=0 -> actions=normal
        match1 = parser.OFPMatch(in_port=9)
        actions1 = [parser.OFPActionSetQueue(0), parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        inst1 = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions1)]
        mod1 = parser.OFPFlowMod(datapath=datapath, priority=200, match=match1, instructions=inst1)
        datapath.send_msg(mod1)

        # in_port=4 → queue=1 -> actions=normal
        match2 = parser.OFPMatch(in_port=4)
        actions2 = [parser.OFPActionSetQueue(1), parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        inst2 = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions2)]
        mod2 = parser.OFPFlowMod(datapath=datapath, priority=200, match=match2, instructions=inst2)
        datapath.send_msg(mod2)

        self.logger.info("Queue flows installed: in_port 9 -> Q10, in_port 4 -> Q20")


    def add_flow(self, datapath, match, actions):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath,
                                priority=300,
                                match=match,
                                instructions=inst)
        datapath.send_msg(mod)

    def send_to_controller_from_port(self, datapath, port_no):
        """ Adds flow which sends all packets from the switch port given to the controller """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch(in_port=port_no)
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                           ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(datapath=datapath,
                                priority=250,
                                match=match,
                                instructions=inst)
        datapath.send_msg(mod)
        
    def set_ovsdb_addr(self, dpid):
        """ Sets OVSDB address"""
        url = "http://localhost:8080/v1.0/conf/switches/"+dpid+"/ovsdb_addr"

        data = json.dumps("tcp:192.168.230.5:6632")
        headers = {"Content-Type": "application/json"}
        try:
            r = requests.put(url, data=data, headers=headers)
            if r.status_code == 201:
                self.logger.info(f"OVSDB_ADDR set successfully ")
            else:
                self.logger.error(f"Error setting OVSDB_ADDR")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")

    # TODO: Parametrizar port_name
    def create_queues(self, dpid, rate0, rate1):
        """ Creates queues """
        url = "http://localhost:8080/qos/queue/"+dpid 
        
        queues = {
            "port_name": "ens23",
            "type": "linux-htb",
            "queues": [
                {"max_rate": str(rate0)},   # Queue 0
                {"max_rate": str(rate1)},    # Queue 1
                {"max_rate": str(100000000)},   # Queue 2
                {"max_rate": str(1000000000)},   # Queue 3
                {"max_rate": str(3000000000)},   # Queue 4
                {"max_rate": str(4000000000)}   # Queue 5
            ]
        }
        headers = {"Content-Type": "application/json"}
        data = json.dumps(queues)
        try:
            r = requests.post(url, data=data, headers=headers)
            response = json.loads(r.text)
            if r.status_code == 200 and response[0]["command_result"]["result"] == "success":
                self.logger.info(f"QoS queues created successfully")
            else:
                self.logger.error(f"Error creating QoS: {r.text}")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")

    def get_queues(self, dpid):
        """ Returns queues of the switch"""
        url = "http://localhost:8080/qos/queue/"+dpid 
        
        headers = {"Content-Type": "application/json"}
        try:
            r = requests.get(url, headers=headers)
            response = json.loads(r.text)
            if r.status_code == 200 :
                return response
            else:
                self.logger.error(f"Error getting QoS: {r.text}")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")
            
    def update_queues(self, dpid, rate0):
        """ Updates queues with a given rate """
        url = "http://localhost:8080/qos/queue/"+dpid 
        
        queues = {
            "port_name": "ens23",
            "type": "linux-htb",
            "queues": [
                {"max_rate": str(rate0)}    # Queue 0
                #{"max_rate": str(rate1)}   # Queue 1
                ]
        }
        headers = {"Content-Type": "application/json"}
        data = json.dumps(queues)
        try:
            r = requests.post(url, data=data, headers=headers)
            json.loads(r.text)
            
            if r.status_code == 200:
                self.logger.info(f"QoS queues updated successfully")
            else:
                self.logger.error(f"Error updating QoS: {r.text}")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")
            
    def prediction_throughput_handler(self):
        # Downlink throughput
        for ip in self.ip_map:
            throughput = (int(self.ip_map[ip]['bytes']) * 8) / 1000000 / 0.5  # UE throughput in Mbps
            self.logger.info("%s throughput: %.2f Mbps",ip, throughput)
            self.ip_map[ip]['bytes'] = 0
            
            self.window[ip].append(float(throughput))

            if len(self.window[ip]) == LOOKBACK:
                aux_window = self.window[ip]
                input_data = np.array(aux_window).reshape(1, LOOKBACK, 1)
                throughput_pred, class_pred = self.model.predict(input_data)

                throughput = throughput_pred[0][0]  # Throughput prediction
                class_value = np.argmax(class_pred[0])  # Class prediction

                class_ = self.get_class(class_value)
                self.logger.info(f"%s throughput prediction: {throughput:.2f}", ip)
                self.logger.info(f"%s class prediction: {class_}", ip) 

                self.last_pred[ip].append(class_value)
                if len(self.last_pred[ip]) == LAST_PRED:
                    max_class = max(set(self.last_pred[ip]), key=self.last_pred[ip].count)
                    self.add_service_queue_flow(self.datapath, self.ip_map[ip]['tun_id'], 9, 2) # 9 = UPF Download port
                    self.last_pred[ip].clear()
                    
                #self.qos_policy(throughput)
                print("\n")
        threading.Timer(0.5, self.prediction_throughput_handler).start()
    
    def get_class(self, class_value):
        if class_value == 0:
            return 'youtube'
        elif class_value == 1:
            return 'twitch'
        elif class_value == 2:
            return 'prime'
        elif class_value == 3:
            return 'tiktok'
            
    def qos_policy(self, prediction):
        """ Policy to establish the new max_rate of the queues """
        response_queues = self.get_queues(self.dpid_str)
        curr_max_rate = int(response_queues[0]["command_result"]["details"]["ens23"]["0"]["config"]["max-rate"])
        prediction = prediction * 1000000
        print(prediction, curr_max_rate)
        if prediction > curr_max_rate: 
            max_rate = int(prediction)
            self.update_queues(self.dpid_str, max_rate)
        elif curr_max_rate != 10000000:
            self.update_queues(self.dpid_str, 10000000) # 10 Mbps

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _get_inner_ip(self, ev):
        """ Parses the raw packets coming from the switch to get the UEs IP """
        msg = ev.msg
        #dp = msg.datapath
        #port = msg.match['in_port']

        # Parse the raw packet
        pkt = packet.Packet(msg.data)
        pkt.get_protocol(ethernet.ethernet)
        ip = pkt.get_protocol(ipv4.ipv4)
        udp_pkt = pkt.get_protocol(udp.udp)
    
        if ip and udp_pkt and udp_pkt.dst_port == 2152:
            #self.logger.info("GTP packet detected from %s", ip.src)

            # Calc UDP payload offset 
            eth_length = 14  # Ethernet header is always 14 bytes
            ip_length = (ip.header_length) * 4
            udp_length = 8  # UDP header is 8 bytes

            gtp_offset = eth_length + ip_length + udp_length
            gtp_payload = msg.data[gtp_offset:]
            # GTP header is 16 bytes
            if len(gtp_payload) > 16:
                try:
                    gtp_header = gtp_payload[:16]
                    # tunnel_id
                    teid = struct.unpack_from("!I", gtp_header, 4)[0]
                    
                    ip_inner = scapy_IP(gtp_payload[16:])
    
                    #UEs IP list
                    if ip_inner.dst not in self.ues:
                        self.ues.add(ip_inner.dst)

                    # [IP] {bytes, tun_uid} map    
                    if ip_inner.dst in self.ip_map:
                        self.ip_map[ip_inner.dst]['bytes'] += len(msg.data) # Whole package length
                    else:
                       
                        print(teid)
                        self.ip_map[ip_inner.dst] = {'bytes':len(msg.data), 'tun_id':teid}
                        print(self.ip_map)
                except Exception as e:
                    self.logger.warning("UE IP could not be accesed: %s", e)

    def add_service_queue_flow(self, datapath, tun_id, in_port, queue_id):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        padding = 10
        tun_id = hex(f"{teid:#0{padding}x}")
        match = parser.OFPMatch(
            in_port=in_port,
            eth_type=0x0800,         # IPv4
            ip_proto=17,             # UDP
            #udp_dst=2152,            # GTP-U Port
            tunnel_id=tun_id         # TEID match
        )

        # Actions, (setQueue and normal)
        actions = [
            parser.OFPActionSetQueue(queue_id),
            parser.OFPActionOutput(ofproto.OFPP_NORMAL)
        ]

        # Install flow
        self.add_flow(datapath, match=match, actions=actions)

        class_ = self.get_class(queue_id-2)
        ip = [ip for ip in self.ip_map if self.ip_map[ip]['tun_id'] == tun_id]
        self.logger.info("Flow installed on queue: %s(%s) for UE: %s(%s)",class_,queue_id, ip, tun_id)
        
    def monitor_throughput(self, interval):
        for ip, (bytes_, tun_id) in self.ip_map.items():
            throughput = (int(bytes_) * 8) / 1000000 / interval  #Mbps
            
            self.logger.info("%s throughput: %.2f Mbps",ip, throughput)

            self.ip_map[ip]['bytes'] = 0
        
        # Call again after <interval> seconds
        threading.Timer(interval, self.monitor_throughput, [interval]).start()