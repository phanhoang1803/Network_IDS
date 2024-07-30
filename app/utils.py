import pandas as pd
from scapy.all import rdpcap, TCP, IP, IPv6, UDP, ICMP, ARP

from DetectorModels.src.data.data_processing import generate_features, process_data
from app import lgbm_model

def lgbm_inference(df_origin, encoder, scaler):
    df = df_origin.copy()
    
    df = generate_features(df)
    df, _, _ = process_data(df, encoder, scaler)

    # Drop label column if it exists
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Drop
    # drop_cols = ["rate", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"]
    # df = df.drop(columns=drop_cols)

    # Predict intrusion
    prediction = lgbm_model.predict(df, predict_disable_shape_check=True)
    print(prediction)
    # Convert prediction to intrusion boolean
    is_intrusion = (prediction > 0.59).astype(int)
    
    return is_intrusion

def get_service(port):
    services = {
        80: 'http',
        21: 'ftp',
        25: 'smtp',
        22: 'ssh',
        53: 'dns',
        20: 'ftp-data',
        194: 'irc',
        443: 'https',
        110: 'pop3',
        995: 'pop3s',
        143: 'imap',
        993: 'imaps'
    }
    return services.get(port, '-')

def get_state(tcp_flags):
    # Dictionary to map TCP flag combinations to state names
    state_mapping = {
        'F': 'FIN',     # Finish
        'S': 'SYN',     # Synchronize
        'R': 'RST',     # Reset
        'P': 'PUSH',    # Push
        'A': 'ACK',     # Acknowledge to FIN, for more synchronize with dataset, we don't use ACK, so we use FIN
        'U': 'URG',     # Urgent
        'E': 'ECE',     # ECN-Echo
        'C': 'CWR',     # Congestion Window Reduced
        
        # For combinations with multiple flags, we just use the first one
        'FS': 'FIN', 
        'FA': 'FIN',
        'SA': 'SYN-ACK',
        'RA': 'RST',
        'PA': 'PUSH',
        'UA': 'URG',
        'EA': 'ECE',
        'CA': 'CWR', 
    }

    # Map the flags to the corresponding state
    return state_mapping.get(str(tcp_flags), '-')

def get_ip_layer(packet):
    for ip in (IP, IPv6):
        if ip in packet:
            return packet[ip]
    return None

def get_proto_layer(packet):
    for proto in (TCP, UDP, ICMP, ARP):
        if proto in packet:
            return packet[proto]
    return None

def parse_proto(proto_num):
    protocols = {
        1: 'icmp',
        6: 'tcp',
        17: 'udp',
        0x0806: 'arp',
    }
    return protocols.get(proto_num, '-')

def is_http_request(packet):
    try:
        # Extract payload from packet
        payload = packet.payload
        if not payload:
            return False
        
        # Convert payload to string and check for HTTP request patterns
        payload_str = str(payload, 'utf-8', errors='ignore')
        
        # Check if the payload starts with a common HTTP request method
        request_methods = ['GET ', 'POST ', 'PUT ', 'DELETE ', 'HEAD ', 'OPTIONS ', 'PATCH ']
        return any(payload_str.startswith(method) for method in request_methods)
    
    except Exception as e:
        print(f"Error checking HTTP request: {e}")
        return False

def is_http_response(packet):
    try:
        # Extract payload from packet
        payload = packet.payload
        if not payload:
            return False
        
        # Convert payload to string and check for HTTP response patterns
        payload_str = str(payload, 'utf-8', errors='ignore')
        
        # Check if the payload starts with a common HTTP response status line
        response_status_prefixes = ['HTTP/', 'HTTP/1.0 ', 'HTTP/1.1 ', 'HTTP/2.0 ']
        return any(payload_str.startswith(prefix) for prefix in response_status_prefixes)
    
    except Exception as e:
        print(f"Error checking HTTP response: {e}")
        return False


def initialize_connections(packets):
    connections = {}
    
    for packet in packets:
        try:
            ip_layer = get_ip_layer(packet)
            pro_layer = get_proto_layer(packet)
            
            if ip_layer is None or pro_layer is None:
                continue
            
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            src_port = pro_layer.sport
            dst_port = pro_layer.dport
            
            proto = parse_proto(ip_layer.proto)
            if proto == 'udp':
                state = 'INT'
            else:
                state = get_state(pro_layer.flags) if hasattr(pro_layer, 'flags') else '-'
                
            service = get_service(dst_port)
            
            ttl = ip_layer.ttl if hasattr(ip_layer, 'ttl') else 0
            
            swin = pro_layer.window if hasattr(pro_layer, 'window') else 0
            stcpb = pro_layer.seq if hasattr(pro_layer, 'seq') else 0
            
            timestamp = packet.time
            
            conn_key = (src_ip, dst_ip, src_port, dst_port, proto)
            rev_conn_key = (dst_ip, src_ip, dst_port, src_port, proto)
            
            if conn_key not in connections:
                connections[conn_key] = {
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    
                    
                    'start_time': timestamp,
                    'end_time': timestamp,
                    
                    'proto': proto,
                    'state': state,
                    'service': service,
                    
                    'spkts': 0,             # Source to destination packet count 
                    'dpkts': 0,             # Destination to source packet count
                    
                    'sbytes': 0,            # Source to destination transaction bytes 
                    'dbytes': 0,            # Destination to source transaction bytes
                    
                    'rate': 0.0,
                    
                    'sttl': ttl,            # Source to destination time to live value 
                    'dttl': 0,              # Destination to source time to live value
                    
                    'sload': 0.0,             # Source bits per second
                    'dload': 0.0,             # Destination bits per second
                    
                    'sloss': 0,             # Source packets retransmitted or dropped 
                    'dloss': 0,             # Destination packets retransmitted or dropped
                    
                    'sinpkt': 0.0,           # Source interpacket arrival time (mSec)
                    'dinpkt': 0.0,           # Destination interpacket arrival time (mSec)
                    
                    'sjit': 0.0,              # Source jitter (mSec)
                    'djit': 0.0,              # Destination jitter (mSec)
                    
                    'swin': swin,           # Source TCP window advertisement value
                    'stcpb': stcpb,         # Source TCP base sequence number
                    'dtcpb': 0,         # Destination TCP base sequence number
                    'dwin': 0,              # Destination TCP window advertisement value
                    
                    'tcprtt': 0.0,            # TCP connection setup round-trip time, the sum of ’synack’ and ’ackdat’.
                    'synack': 0.0,            # TCP connection setup time, the time between the SYN and the SYN_ACK packets.
                    'ackdat': 0.0,            # TCP connection setup time, the time between the SYN_ACK and the ACK packets.
                    
                    'smean': 0,             # Mean of the ?ow packet size transmitted by the src 
                    'dmean': 0,             # Mean of the ?ow packet size transmitted by the dst 

                    'trans_depth': 0,       # Represents the pipelined depth into the connection of http request/response transaction
                    'response_body_len': 0, # Actual uncompressed content size of the data transferred from the server’s http service.
                    
                    'ct_srv_src': 0,        # No. of connections that contain the same service (14) and source address (1) in 100 connections according to the last time (26).
                    'ct_state_ttl': 0,      # No. for each state (6) according to specific range of values for source/destination time to live (10) (11).
                    'ct_dst_ltm': 0,        # No. of connections of the same destination address (3) in 100 connections according to the last time (26).
                    'ct_src_dport_ltm': 0,  # No of connections of the same source address (1) and the destination port (4) in 100 connections according to the last time (26).
                    'ct_dst_sport_ltm': 0,  # No of connections of the same destination address (3) and the source port (2) in 100 connections according to the last time (26).
                    'ct_dst_src_ltm': 0,    # No of connections of the same source (1) and the destination (3) address in in 100 connections according to the last time (26).
                    'is_ftp_login': 0,      # If the ftp session is accessed by user and password then 1 else 0. 
                    'ct_ftp_cmd': 0,        # No of flows that has a command in ftp session.
                    'ct_flw_http_mthd': 0,  # No. of flows that has methods such as Get and Post in http service.
                    'ct_src_ltm': 0,        # No. of connections of the same source address (1) in 100 connections according to the last time (26).
                    'ct_srv_dst': 0,        # No. of connections that contain the same service (14) and destination address (3) in 100 connections according to the last time (26).

                    'is_sm_ips_ports': 0,    # If source (1) and destination (3)IP addresses equal and port numbers (2)(4)  equal then, this variable takes value 1 else 0
                    
                    
                    
                    
                    
                    
                    
                    'http_requests': 0,  # Track HTTP requests
                    'http_responses': 0, # Track HTTP responses
                }
            
            conn = connections[conn_key]
            conn['end_time'] = timestamp
            
            # TODO: Basic features
            conn['state'] = state
            conn['service'] = service
            conn['spkts'] += 1
            conn['sbytes'] += len(packet)
            conn['sttl'] = ttl
            conn['sload'] = (conn['sbytes'] * 8.0) / float(conn['end_time'] - conn['start_time'] + 1e-6)
            
            # sloss
            if proto == 'tcp' and hasattr(pro_layer, 'flags') and pro_layer.flags == 'R':
                conn['sloss'] += 1
            
            # TODO: Content features
            conn['swin'] = swin
            conn['stcpb'] = stcpb
            conn['smean'] = int(conn['sbytes'] / (conn['spkts'] + 1e-6))
            
            # trans_depth
            if proto == 'tcp' and service == 'http':
                if is_http_request(packet):
                    conn['http_requests'] += 1
                elif is_http_response(packet):
                    conn['http_responses'] += 1
            conn['trans_depth'] = conn['http_requests'] - conn['http_responses']
            
            # response_body_len
            if proto == 'tcp' and service == 'HTTP':
                if hasattr(pro_layer, 'http_body_length'):
                    conn['response_body_len'] += pro_layer.http_body_length
            
            # TODO: Time features
            # sinpkt
            if 'last_timestamp' in conn:
                conn['sinpkt'] = (timestamp - conn['last_timestamp']) * 1000  # Convert to milliseconds
            else:
                conn['sinpkt'] = 0
            conn['last_timestamp'] = timestamp
            
            # sjit
            if 'arrival_times' not in conn:
                conn['arrival_times'] = []
            if 'last_arrival' in conn:
                interarrival_time = (timestamp - conn['last_arrival']) * 1000  # Convert to milliseconds
                conn['arrival_times'].append(interarrival_time)
                if len(conn['arrival_times']) > 1:
                    mean_interarrival = sum(conn['arrival_times']) / len(conn['arrival_times'])
                    variance = sum((x - mean_interarrival) ** 2 for x in conn['arrival_times']) / len(conn['arrival_times'])
                    conn['sjit'] = variance ** 0.5  # Jitter is the standard deviation of interarrival times
                else:
                    conn['sjit'] = 0
            else:
                conn['sjit'] = 0
            conn['last_arrival'] = timestamp
            
            # synack, ackdat, tcprtt
            if proto == 'tcp' and state == 'SYN':
                conn['syn_time'] = timestamp
            elif proto == 'tcp' and state == 'SYN-ACK':
                conn['synack_time'] = timestamp
            elif proto == 'tcp' and state == 'ACK':
                conn['ack_time'] = timestamp
            
            
            # TODO: General purpose features
            # ct_state_ttl (Count of connections with same state and TTL)
            
            # ct_flw_http_mthd (Count of flows that has methods such as Get and Post in http service)
            if proto == 'tcp' and service == 'http':
                print(dir(pro_layer))
                if hasattr(pro_layer, 'http_method'):
                    conn['ct_flw_http_mthd'] += 1
                    
            # is_ftp_login (If the ftp session is accessed by user and password then 1 else 0)
            if proto == 'tcp' and service == 'FTP':
                if hasattr(pro_layer, 'ftp_command') and 'USER' in pro_layer.ftp_command:
                    conn['is_ftp_login'] = 1
            # ct_ftp_cmd (Count of flows that has a command in ftp session)
            if proto == 'tcp' and service == 'FTP':
                if hasattr(pro_layer, 'ftp_command'):
                    conn['ct_ftp_cmd'] += 1
                    
            # is_sm_ips_ports (If source (1) and destination (3)IP addresses equal and port numbers (2)(4)  equal then, this variable takes value 1 else 0)
            conn['is_sm_ips_ports'] = int(src_ip == dst_ip and src_port == dst_port)

            # TODO: Connection Counts (ct_* metrics)
            # ct_srv_src: Count of connections with same service and source IP in 100 connections according to the last time (26).
            
            # ct_state_ttl: Count of connections with same state and TTL
            
            # ct_dst_ltm: Count of connections of the same destination address (3) in 100 connections according to the last time (26).
            
            # ct_src_dport_ltm: Count of connections of the same source address (1) and the destination port (4) in 100 connections according to the last time (26).
            
            # ct_dst_sport_ltm: Count of connections of the same destination address (3) and the source port (2) in 100 connections according to the last time (26).
            
            # ct_dst_src_ltm: Count of connections of the same source (1) and the destination (3) address in in 100 connections according to the last time (26).
            
            # ct_srv_dst: Count of connections that contain the same service (14) and destination address (3) in 100 connections according to the last time (26).

        except Exception as e:
            print(f"Error processing packet: {packet}")
            print(f"Error initializing connection: {e}")
            continue

    return connections

def aggregate_connections(connections):
    from collections import defaultdict, deque
    
    seen_connections = set()
    aggregated_connections = {}
    
    # Initialize counters for connection counts with sliding window
    sliding_window_size = 100
    window_connections = {
        'srv_src': deque(maxlen=sliding_window_size),
        'state_ttl': deque(maxlen=sliding_window_size),
        'dst_ltm': deque(maxlen=sliding_window_size),
        'src_dport_ltm': deque(maxlen=sliding_window_size),
        'dst_sport_ltm': deque(maxlen=sliding_window_size),
        'dst_src_ltm': deque(maxlen=sliding_window_size),
        'srv_dst': deque(maxlen=sliding_window_size),
    }
    
    # Initialize counters
    counters = {
        'srv_src': defaultdict(int),
        'state_ttl': defaultdict(int),
        'dst_ltm': defaultdict(int),
        'src_dport_ltm': defaultdict(int),
        'dst_sport_ltm': defaultdict(int),
        'dst_src_ltm': defaultdict(int),
        'srv_dst': defaultdict(int),
    }
    
    for conn_key, conn in connections.items():
        src_ip, dst_ip, src_port, dst_port, proto = conn_key
        rev_conn_key = (dst_ip, src_ip, dst_port, src_port, proto)
        
        if rev_conn_key not in connections:
            continue
        
        if rev_conn_key not in seen_connections:
            seen_connections.add(conn_key)
            rev_conn = connections[rev_conn_key]
            
            # Update destination connection statistics
            conn['end_time'] = rev_conn['end_time']
            conn['dpkts'] = rev_conn['spkts']
            conn['dbytes'] = rev_conn['sbytes']
            
            total_bytes = conn['sbytes'] + conn['dbytes']
            conn['rate'] = total_bytes / (conn['end_time'] - conn['start_time'])
            
            # conn['rate'] = 0 # TODO: calculate rate (The paper doesn't specify this, but it's in the data)
            
            conn['dttl'] = rev_conn['sttl']
            conn['dload'] = rev_conn['sload']
            conn['dloss'] = rev_conn['sloss']
            conn['dinpkt'] = rev_conn['sinpkt']
            conn['djit'] = rev_conn['sjit']
            conn['dwin'] = rev_conn['swin']
            conn['dtcpb'] = rev_conn['stcpb']
        
            # A --> B : SYN
            # B --> A: SYN-ACK
            # A --> B: ACK
            # So syn_time, and ack_time belong to A --> B
            #              synack_time belongs to B --> A
            conn['synack'] = rev_conn['synack_time'] - conn['syn_time'] if 'synack_time' in rev_conn and 'syn_time' in conn else 0
            conn['ackdat'] = conn['ack_time'] - rev_conn['synack_time'] if 'ack_time' in conn and 'synack_time' in rev_conn else 0
            conn['tcprtt'] = conn['synack'] + conn['ackdat']
            
            conn['dmean'] = int(conn['dbytes'] / (conn['dpkts'] + 1e-6))
            
    
            # TODO: Connection Counts (ct_* metrics)
            # Add the current connection to the sliding windows
            window_connections['srv_src'].append((conn['service'], src_ip)) 
            window_connections['state_ttl'].append((proto, conn.get('state', ''), conn.get('ttl', '')))
            window_connections['dst_ltm'].append((dst_ip,))
            window_connections['src_dport_ltm'].append((src_ip, dst_port))
            window_connections['dst_sport_ltm'].append((dst_ip, src_port))
            window_connections['dst_src_ltm'].append((dst_ip, src_ip))
            window_connections['srv_dst'].append((conn['service'], dst_ip))
            
            # Update counters based on the sliding windows
            for metric, window in window_connections.items():
                counters[metric].clear()  # Reset counters for the current window
                for item in window:
                    counters[metric][item] += 1
            
            # Add metrics to connection data
            conn['ct_srv_src'] = counters['srv_src'][(conn['service'], src_ip)]
            conn['ct_state_ttl'] = counters['state_ttl'][(proto, conn.get('state', ''), conn.get('ttl', ''))]
            conn['ct_dst_ltm'] = counters['dst_ltm'][(dst_ip,)]
            conn['ct_src_dport_ltm'] = counters['src_dport_ltm'][(src_ip, dst_port)]
            conn['ct_dst_sport_ltm'] = counters['dst_sport_ltm'][(dst_ip, src_port)]
            conn['ct_dst_src_ltm'] = counters['dst_src_ltm'][(dst_ip, src_ip)]
            conn['ct_srv_dst'] = counters['srv_dst'][(conn['service'], dst_ip)]
            
            # ct_srv_src: Count of connections with same service and source IP in 100 connections according to the last time.
            
            # ct_state_ttl: No. for each state according to specific range of values for source/destination time to live.
            
            # ct_dst_ltm: Count of connections of the same destination address in 100 connections according to the last time.
            
            # ct_src_dport_ltm: Count of connections of the same source address and the destination port in 100 connections according to the last time.
            
            # ct_dst_sport_ltm: Count of connections of the same destination address and the source port in 100 connections according to the last time.
            
            # ct_dst_src_ltm: Count of connections of the same source and the destination address in in 100 connections according to the last time.
            
            # ct_srv_dst: Count of connections that contain the same service and destination address in 100 connections according to the last time.
            
            aggregated_connections[conn_key] = conn
    
    return aggregated_connections

def parse_pcap(file):
    """
    Parses a pcap file and returns a pandas DataFrame of connection information.

    Args:
        file (str): Path to the pcap file.

    Returns:
        pandas.DataFrame: DataFrame containing connection information.
            Columns:
                - dur (float): Duration of the connection.
                - proto (str): Protocol of the connection.
                - service (str): Service of the connection.
                - state (str): State of the connection.
                - spkts (int): Number of packets sent by the source.
                - dpkts (int): Number of packets received by the destination.
                - sbytes (int): Number of bytes sent by the source.
                - dbytes (int): Number of bytes received by the destination.
                - rate (int): Rate of the connection.
                - sttl (int): Time to live of the source.
                - dttl (int): Time to live of the destination.
                - sload (int): Load of the source.
                - dload (int): Load of the destination.
                - sloss (int): Loss of the source.
                - dloss (int): Loss of the destination.
                - sinpkt (int): Number of packets sent by the source within the connection.
                - dinpkt (int): Number of packets received by the destination within the connection.
                - sjit (int): Jitter of the source.
                - djit (int): Jitter of the destination.
                - swin (int): Window size of the source.
                - stcpb (int): TCP buffer size of the source.
                - dtcpb (int): TCP buffer size of the destination.
                - dwin (int): Window size of the destination.
                - tcprtt (int): Round trip time of the TCP connection.
                - synack (int): Time taken for SYN-ACK handshake.
                - ackdat (int): Time taken for ACK-data handshake.
                - smean (int): Mean bytes per packet sent by the source.
                - dmean (int): Mean bytes per packet received by the destination.
                - trans_depth (int): Transmission depth of the connection.
                - response_body_len (int): Length of the response body.
                - ct_state_ttl (int): State TTL of the connection.
                - is_ftp_login (bool): Whether the connection is an FTP login.
                - ct_ftp_cmd (int): FTP command of the connection.
                - ct_flw_http_mthd (int): HTTP method of the connection.
                - is_sm_ips_ports (bool): Whether the connection involves IPs and ports in a suspicious manner.
    """
    print("[PARSE_PCAP] Parsing pcap file...")
    packets = rdpcap(file)
    data = []
    print("[PARSE_PCAP] Initializing connections...")
    connections = initialize_connections(packets)
    
    print("[PARSE_PCAP] Aggregating connections...")
    aggregated_connections = aggregate_connections(connections)
            
    data = []
    print("[PARSE_PCAP] Parsing connections...")
    for conn_key, conn in aggregated_connections.items():
        src_ip, dst_ip, src_port, dst_port, proto = conn_key
        dur = conn['end_time'] - conn['start_time']
        data.append([
            float(dur), 
            conn['proto'], 
            conn['service'], 
            conn['state'], 
            int(conn['spkts']), 
            int(conn['dpkts']), 
            int(conn['sbytes']), 
            int(conn['dbytes']),
            float(conn['rate']), 
            int(conn['sttl']), 
            int(conn['dttl']), 
            float(conn['sload']), 
            float(conn['dload']), 
            int(conn['sloss']), 
            int(conn['dloss']), 
            float(conn['sinpkt']),
            float(conn['dinpkt']), 
            float(conn['sjit']), 
            float(conn['djit']), 
            int(conn['swin']), 
            int(conn['stcpb']), 
            int(conn['dtcpb']), 
            int(conn['dwin']), 
            float(conn['tcprtt']),
            float(conn['synack']), 
            float(conn['ackdat']), 
            int(conn['smean']), 
            int(conn['dmean']), 
            int(conn['trans_depth']), 
            int(conn['response_body_len']),
            int(conn['ct_srv_src']),
            int(conn['ct_state_ttl']), 
            int(conn['ct_dst_ltm']), 
            int(conn['ct_src_dport_ltm']),
            int(conn['ct_dst_sport_ltm']),
            int(conn['ct_dst_src_ltm']), 
            int(conn['is_ftp_login']),
            int(conn['ct_ftp_cmd']),
            int(conn['ct_flw_http_mthd']), 
            int(conn['ct_src_ltm']), 
            int(conn['ct_srv_dst']), 
            int(conn['is_sm_ips_ports']),
        ])

    return pd.DataFrame(data, columns=[
        'dur', 
        'proto', 
        'service', 
        'state', 
        'spkts', 
        'dpkts', 
        'sbytes', 
        'dbytes',
        'rate', 
        'sttl', 
        'dttl', 
        'sload', 
        'dload', 
        'sloss', 
        'dloss', 
        'sinpkt',
        'dinpkt', 
        'sjit', 
        'djit', 
        'swin', 
        'stcpb', 
        'dtcpb', 
        'dwin', 
        'tcprtt',
        'synack', 
        'ackdat', 
        'smean', 
        'dmean', 
        'trans_depth', 
        'response_body_len',
        'ct_srv_src', 
        'ct_state_ttl', 
        'ct_dst_ltm', 
        'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 
        'ct_dst_src_ltm', 
        'is_ftp_login', 
        'ct_ftp_cmd',
        'ct_flw_http_mthd', 
        'ct_src_ltm',
        'ct_srv_dst', 
        'is_sm_ips_ports',
    ])
