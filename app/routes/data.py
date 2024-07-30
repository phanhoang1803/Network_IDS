# file: routes/data.py

from flask import Blueprint, jsonify, request
from flask_socketio import emit
import os
import threading
import time
import pandas as pd
from scapy.all import sniff, wrpcap
from tqdm import tqdm
from utils import parse_pcap, lgbm_inference
from app import socketio, lgbm_model, encoder, scaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_bp = Blueprint("data", __name__)

# Directory to save pcap files
PCAP_DIR = 'pcap_files'
os.makedirs(PCAP_DIR, exist_ok=True)

# Global variables for monitoring control
monitoring_thread = None
stop_monitoring = threading.Event()
sniffed_packets = []
duration = 60  # Default duration in seconds for each epoch

def countdown_timer(interval):
    for i in range(int(interval), 0, -1):
        if stop_monitoring.is_set():
            break
        print(f"Time remaining: {i} seconds", end="\r")
        time.sleep(1)
    print(" " * 30, end="\r")  # Clear the line

def progress_bar(interval):
    with tqdm(total=interval, desc="Sniffing", bar_format="{desc}: {bar} {remaining}") as pbar:
        for _ in range(int(interval)):
            if stop_monitoring.is_set():
                break
            time.sleep(1)
            pbar.update(1)

def monitor_network(pcap_file, interval):
    global sniffed_packets

    while not stop_monitoring.is_set():
        # Clear previous packets
        sniffed_packets = []

        def packet_handler(packet):
            sniffed_packets.append(packet)
        
        try:
            print("\n[MONITOR NETWORK] Sniffing...\n")
            # Start the progress bar in a separate thread
            progress_thread = threading.Thread(target=progress_bar, args=(interval,))
            progress_thread.start()
            sniff(prn=packet_handler, timeout=interval, stop_filter=lambda x: stop_monitoring.is_set())
            # Ensure the progress bar thread finishes
            progress_thread.join()
            
        except Exception as e:
            print(f"Error during sniffing: {e}")
            continue
        
        try:
            print("\n[MONITOR NETWORK] Saving pcap file...\n")
            wrpcap(pcap_file, sniffed_packets)
        except Exception as e:
            print(f"Error saving pcap file: {e}")
            continue

        # Process and send results back
        try:
            print("\n[MONITOR NETWORK] Parsing pcap file...\n")
            df = parse_pcap(pcap_file)

            print("\n[MONITOR NETWORK] Predicting intrusion...")
            is_intrusion = lgbm_inference(pd.concat([df, pd.DataFrame({"label": [0] * len(df)})], axis=1), encoder, scaler)
            df_result = pd.concat([df, pd.DataFrame({'is_intrusion': is_intrusion})], axis=1).to_json(orient='records')

            socketio.emit('monitor_result', df_result)
            
            df.to_csv('output.csv', index=False)
        except Exception as e:
            print(f"Error processing data: {e}")
        
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


# WebSocket events for detecting disconnections
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")
    stop_monitoring.set()  # Signal the monitoring thread to stop

# To ensure the monitoring thread stops if the application shuts down
def cleanup():
    stop_monitoring.set()
    if monitoring_thread and monitoring_thread.is_alive():
        monitoring_thread.join()
        
import atexit
atexit.register(cleanup)