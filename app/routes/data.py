# file: routes/data.py

from flask import Blueprint, jsonify, request
from flask_socketio import emit
import os
import threading
import time
import pandas as pd
from scapy.all import sniff, wrpcap
from utils import parse_pcap, lgbm_inference
from app import socketio, lgbm_model, encoder, scaler

data_bp = Blueprint("data", __name__)

# Directory to save pcap files
PCAP_DIR = 'pcap_files'
os.makedirs(PCAP_DIR, exist_ok=True)

# Global variables for monitoring control
monitoring_thread = None
stop_monitoring = threading.Event()
sniffed_packets = []
duration = 60  # Default duration in seconds for each epoch

def monitor_network(pcap_file, interval):
    global sniffed_packets

    while not stop_monitoring.is_set():
        # Clear previous packets
        sniffed_packets = []

        def packet_handler(packet):
            sniffed_packets.append(packet)
        
        # Sniff packets for the interval duration
        sniff(prn=packet_handler, timeout=interval, stop_filter=lambda x: stop_monitoring.is_set())
        
        # Save packets to pcap file
        wrpcap(pcap_file, sniffed_packets)

        # Process and send results back
        try:
            df = parse_pcap(pcap_file)
        except Exception as e:
            print(f"Error parsing pcap file: {e}")
            continue

        result = df.to_json(orient='records')
        print(df.info())
        is_intrusion = lgbm_inference(pd.concat([df, pd.DataFrame({"label": [0] * len(df)})], axis=1), encoder, scaler)
        df_result = pd.concat([df, pd.DataFrame({'is_intrusion': is_intrusion})], axis=1).to_json(orient='records')
        
        socketio.emit('monitor_result', df_result)
        
        df.to_csv('output.csv', index=False)
        
        # Wait a moment before starting the next epoch if needed
        time.sleep(1)

@data_bp.route('/start_monitor', methods=['POST', 'GET'])
def start_monitor():
    global monitoring_thread, stop_monitoring, duration

    if monitoring_thread and monitoring_thread.is_alive():
        return jsonify({"error": "Monitoring is already running."}), 400
    
    if request.method == 'GET':
        duration = 2  # 2 second
    else:
        duration = request.json.get('duration', 1) * 60  # Duration in seconds
    
    pcap_file = os.path.join(PCAP_DIR, 'monitoring.pcap')

    stop_monitoring.clear()  # Clear the stop event
    monitoring_thread = threading.Thread(target=monitor_network, args=(pcap_file, duration))
    monitoring_thread.start()
    
    return jsonify({"message": "Monitoring started."})

@data_bp.route('/stop_monitor', methods=['POST', 'GET'])
def stop_monitor():
    global stop_monitoring, monitoring_thread
    
    if not monitoring_thread or not monitoring_thread.is_alive():
        return jsonify({"error": "Monitoring is not running."}), 400
    
    stop_monitoring.set()  # Signal the monitoring thread to stop
    monitoring_thread.join()  # Wait for the thread to finish
    
    return jsonify({"message": "Monitoring stopped."})
