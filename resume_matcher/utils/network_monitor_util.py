import psutil
import time
import sys
from datetime import datetime

# Get the PID from command-line argument
if len(sys.argv) < 2:
    print("Please provide the PID as an argument (e.g., python monitor_network.py 20780)")
    sys.exit(1)
PID = int(sys.argv[1])

# Function to get system-wide network I/O counters
def get_io_counters():
    return psutil.net_io_counters()

# Initialize counters
initial_counters = get_io_counters()
initial_sent = initial_counters.bytes_sent
initial_recv = initial_counters.bytes_recv

# Define intervals
interval = 1  # Monitoring interval in seconds
CHECK_INTERVAL = 10  # Check connections every 10 seconds
check_counter = 0

print("Monitoring system-wide network throughput (Ctrl+C to stop)...")

while True:
    time.sleep(interval)
    current_counters = get_io_counters()
    current_sent = current_counters.bytes_sent
    current_recv = current_counters.bytes_recv

    # Calculate send and receive rates
    sent_rate = (current_sent - initial_sent) / interval
    recv_rate = (current_recv - initial_recv) / interval

    # Print network throughput
    print(f"Send Rate: {sent_rate:.2f} B/s | Receive Rate: {recv_rate:.2f} B/s")

    # Periodically validate traffic by checking active connections
    check_counter += 1
    if check_counter >= CHECK_INTERVAL:
        try:
            # Get all network connections
            connections = psutil.net_connections(kind='inet')
            # Filter for connections belonging to our process with ESTABLISHED status
            active_connections = [
                conn for conn in connections
                if conn.pid == PID and conn.status == 'ESTABLISHED'
            ]
            if active_connections:
                for conn in active_connections:
                    if conn.raddr:  # Ensure remote address exists
                        print(f"{datetime.now()} - Active connection to {conn.raddr.ip}:{conn.raddr.port}")
            else:
                print(f"{datetime.now()} - No active connections for PID {PID}")
        except psutil.AccessDenied:
            print(f"{datetime.now()} - Access denied when trying to access connections")
        except Exception as e:
            print(f"{datetime.now()} - Error checking connections: {e}")
        check_counter = 0

    # Update counters for the next iteration
    initial_sent = current_sent
    initial_recv = current_recv