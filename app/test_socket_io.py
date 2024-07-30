import socketio
import requests
import time

# Create a new Socket.IO client instance
sio = socketio.Client()

# Event handler for 'monitor_result'
@sio.event
def monitor_result(data):
    print('Received monitor result:', data)

# Connect to the Flask Socket.IO server
sio.connect('http://127.0.0.1:5000')

# Start monitoring
response = requests.post('http://127.0.0.1:5000/data/start_monitor', json={"duration": 0.1})  # 1 minute duration
if response.status_code == 200:
    print('Monitoring started.')
else:
    print('Failed to start monitoring:', response.text)

# Wait for 5 minutes
time.sleep(500)

# Stop monitoring
response = requests.post('http://127.0.0.1:5000/data/stop_monitor')
if response.status_code == 200:
    print('Monitoring stopped.')
    print('Data:', response.json())
else:
    print('Failed to stop monitoring:', response.text)

# Disconnect the Socket.IO client
sio.disconnect()
