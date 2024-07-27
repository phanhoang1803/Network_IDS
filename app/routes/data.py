import asyncio
import os
import threading
from flask import Blueprint, Response, request, jsonify
import pyshark

data_bp = Blueprint("data", __name__)


captured_packets = []
capture_thread_started = False

def process_packet(packet):
    """ Process packet to extract relevant information for intrusion detection. """
    try:
        packet_info = {
            "timestamp": packet.sniff_time,
            "src_ip": packet.ip.src,
            "dst_ip": packet.ip.dst,
            "protocol": packet.transport_layer,
            "length": int(packet.length)
        }
        
        # Handle TCP packets
        if packet.transport_layer == 'TCP':
            packet_info.update({
                "src_port": int(packet.tcp.srcport),
                "dst_port": int(packet.tcp.dstport),
                "tcp_flags": packet.tcp.flags,
                "payload_size": len(packet.tcp.payload),
                "window_size": int(packet.tcp.window_size_value)
            })
            
            # Additional TCP-specific features
            if hasattr(packet.tcp, 'seq'):
                packet_info["stcpb"] = int(packet.tcp.seq)
            if hasattr(packet.tcp, 'ack'):
                packet_info["ackdat"] = int(packet.tcp.ack)
        
        # Handle UDP packets
        elif packet.transport_layer == 'UDP':
            packet_info.update({
                "src_port": int(packet.udp.srcport),
                "dst_port": int(packet.udp.dstport),
                "payload_size": len(packet.udp.payload)
            })
        
        # Handle ICMP packets
        elif packet.transport_layer == 'ICMP':
            packet_info.update({
                "icmp_type": packet.icmp.type,
                "icmp_code": packet.icmp.code
            })

        # Capture additional fields for all packets
        if hasattr(packet, 'ip'):
            packet_info.update({
                "ttl": int(packet.ip.ttl),
                "ip_flags": packet.ip.flags
            })
        
        if hasattr(packet, 'icmp'):
            packet_info.update({
                "icmp_type": packet.icmp.type,
                "icmp_code": packet.icmp.code
            })
        
        if hasattr(packet, 'dns'):
            packet_info.update({
                "dns_qry_name": packet.dns.qry_name,
                "dns_resp_name": packet.dns.resp_name
            })
        
        return packet_info
    except AttributeError as e:
        # Handle cases where packet does not have expected layers or attributes
        print(f"Packet processing error: {e}")
        return None

def capture_packets(interface):
    global captured_packets
    pyshark.tshark.tshark.tshark_path = 'C:\\Program Files\\Wireshark\\tshark.exe'  # Ensure this path is correct

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    capture = pyshark.LiveCapture(interface=interface)
    for packet in capture.sniff_continuously():
        processed_packet = process_packet(packet)
        captured_packets.append(processed_packet)
        # Keep the list of packets short to avoid memory issues
        if len(captured_packets) > 1000:
            captured_packets.pop(0)


def start_capture_thread():
    global capture_thread_started
    if not capture_thread_started:
        interface = os.getenv("NETWORK_INTERFACE", "Wi-Fi")  # Change 'Wi-Fi' to your WiFi interface name if different
        capture_thread = threading.Thread(target=capture_packets, args=(interface,))
        capture_thread.daemon = True
        capture_thread.start()
        capture_thread_started = True
        
@data_bp.route("/capture", methods=["GET"])
def capture():
    """
    This route starts a new thread to capture packets from the network interface
    specified in the environment variable NETWORK_INTERFACE. It uses the pyshark
    library to capture live packets. The captured packets are stored in the
    global list captured_packets.

    Returns:
        A Response object that generates a server-sent event stream (SSE) of
        captured packets. The SSE is a text/event-stream mimetype.
    """
    start_capture_thread()
    
    def generate():
        """
        This function generates a server-sent event stream (SSE) of captured
        packets. It continuously checks if there are any packets in the
        captured_packets list and yields them one by one.

        Yields:
            A string in the format "data:{packet}\n\n" where packet is a string
            representation of a captured packet.
        """
        while True:
            if captured_packets:
                packet = captured_packets.pop(0)
                # Yield each captured packet as a server-sent event
                yield f"data:{str(packet)}\n\n"
    
    # Return a Response object that generates the server-sent event stream
    return Response(generate(), mimetype="text/event-stream")
