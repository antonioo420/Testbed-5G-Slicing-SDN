from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

import datetime
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
from collections import deque
import pandas as pd
from sklearn.metrics import mean_squared_error
import pdb
import sys
import os
import subprocess, time, re
import json

LOOKBACK = 30
window = deque(maxlen=LOOKBACK)
class SimpleMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor, self).__init__(*args, **kwargs)
        self.model = load_model("./traffic_predictor.h5", custom_objects={'mse': MeanSquaredError()})
        print("Modelo cargado")
        self.datapaths = {}
        self.prev_rx_bytes = {}  # Almacena los RX anteriores
        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """ Handles state changes of the switch """
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[datapath.id] = datapath
            self.add_default_flow(datapath)
            self.add_queue_flows(datapath)
            time.sleep(1)
            self.set_ovsdb_addr()
            time.sleep(1)
            self.create_queues()
            #self.get_queues()
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def add_default_flow(self, datapath):
        """Add flow "actions=normal" to the switch using OpenFlow """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()  # Match all packets
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]  # Actions normal

        # Crear e instalar la regla de flujo
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=100, match=match, instructions=inst)
        datapath.send_msg(mod)

        self.logger.info("Actions normal flow installed")

    def add_queue_flows(self, datapath):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # in_port=9 → queue=10 -> actions=normal
        match1 = parser.OFPMatch(in_port=9)
        actions1 = [parser.OFPActionSetQueue(0), parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        inst1 = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions1)]
        mod1 = parser.OFPFlowMod(datapath=datapath, priority=200, match=match1, instructions=inst1)
        datapath.send_msg(mod1)

        # in_port=4 → queue=20 -> actions=normal
        match2 = parser.OFPMatch(in_port=4)
        actions2 = [parser.OFPActionSetQueue(1), parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        inst2 = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions2)]
        mod2 = parser.OFPFlowMod(datapath=datapath, priority=200, match=match2, instructions=inst2)
        datapath.send_msg(mod2)

        self.logger.info("Queue flows installed: in_port 9 -> Q10, in_port 4 -> Q20")

    def set_ovsdb_addr(self):
        url = "http://localhost:8080/v1.0/conf/switches/0000bc24113d8d35/ovsdb_addr"

        data = json.dumps("tcp:127.0.0.1:6632")
        headers = {"Content-Type": "application/json"}
        try:
            r = requests.put(url, data=data, headers=headers)
            if r.status_code == 201:
                self.logger.info(f"OVSDB_ADDR set successfully ")
            else:
                self.logger.error(f"Error setting OVSDB_ADDR")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")
            
    def create_queues(self):
        url = "http://localhost:8080/qos/queue/0000bc24113d8d35" 
        
        queues = {
            "port_name": "enp2s1",
            "type": "linux-htb",
            "queues": [
                #[{"queue": "10"}, 
                {"max_rate": "90000000"},
                
               # [#{"queue": "20"}, 
                {"max_rate": "20000000"}
            ]
        }
        headers = {"Content-Type": "application/json"}
        data = json.dumps(queues)
        try:
            r = requests.post(url, data=data, headers=headers)
            #print(r.text)
            response = json.loads(r.text)
            if r.status_code == 200 and response[0]["command_result"]["result"] == "success":
                self.logger.info(f"QoS queues created successfully")
            else:
                self.logger.error(f"Error creating QoS: {r.text}")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")

    def get_queues(self):
        url = "http://localhost:8080/qos/queue/0000bc24113d8d35" 
        
        headers = {"Content-Type": "application/json"}
        try:
            r = requests.get(url, headers=headers)
            #print(r.text)
            response = json.loads(r.text)
            if r.status_code == 200 :
                print(r.text)
                self.logger.info(f"QoS queues created successfully")
            else:
                self.logger.error(f"Error creating QoS: {r.text}")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")
            
    def update_queues(self):
        url = "http://localhost:8080/qos/queue/0000bc24113d8d35" 
        
        queues = {
            "port_name": "enp2s1",
            "type": "linux-htb",
            "queues": [
                {"max_rate": "500000"},
                {"min_rate": "800000"}
            ]
        }
        headers = {"Content-Type": "application/json"}
        data = json.dumps(queues)
        try:
            r = requests.post(url, data=data, headers=headers)
            #print(r.text)
            response = json.loads(r.text)
            if r.status_code == 200 and response[0]["command_result"]["result"] == "success":
                self.logger.info(f"QoS queues created successfully")
            else:
                self.logger.error(f"Error creating QoS: {r.text}")
        except Exception as e:
            self.logger.error(f"Exception while requesting to REST API: {e}")
                
    def _monitor(self):
        """ Sends stats request every 0.5 seconds """
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(0.5)  # Intervalo de monitoreo

    def _request_stats(self, datapath):
        """ Requests stats of switch ports """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def getclass(self, class_value):
        if class_value == 0:
            return 'youtube'
        elif class_value == 1:
            return 'twitch'
        elif class_value == 2:
            return 'prime'
        elif class_value == 3:
            return 'tiktok'
        elif class_value == 4:
            return 'navegacion web'

    
"""
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        #  Receives and procceses ports stats
        body = ev.msg.body
        now_rx_bytes = {}

        for stat in sorted(body, key=lambda x: x.port_no):
            port = stat.port_no
            now_rx_bytes[port] = stat.rx_bytes

            if port == 9:
                if port in self.prev_rx_bytes:

                    # Downlink throughput
                    rx_diff = stat.rx_bytes - self.prev_rx_bytes[port]
                    throughput_rx = (rx_diff * 8 * 2) / 1000000 # (Mbps)

                    self.logger.info(f'Port {port}: RX {throughput_rx:.2f} Mbps')

                    window.append(float(throughput_rx))

                    if len(window) == LOOKBACK:
                        input_data = np.array(window).reshape(1, LOOKBACK, 1)
                        throughput_pred, class_pred = self.model.predict(input_data)

                        throughput = throughput_pred[0]  # Throughput prediction
                        class_value = np.argmax(class_pred[0])  # Class prediction

                        class_ = self.getclass(class_value)
                        self.logger.info(f"Predicción throughput: {throughput[0]:.2f}")
                        print(f"Predicción clase: {class_}")
                        #max_rate = int(prediction * 1.2)  # Buffer factor
                        #update_queue_max_rate(max_rate)
        self.prev_rx_bytes = now_rx_bytes"""