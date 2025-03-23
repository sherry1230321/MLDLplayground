import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta

def generate_network_data(num_nodes=50, num_connections=100):
    """
    Generate simulated network data for visualization and analysis.
    
    Args:
        num_nodes: Number of network nodes
        num_connections: Number of connections between nodes
    
    Returns:
        Dictionary containing network data
    """
    # Create nodes with IP addresses and attributes
    nodes = []
    for i in range(num_nodes):
        node_type = random.choice(['server', 'client', 'router', 'firewall', 'database'])
        security_level = random.choice(['high', 'medium', 'low'])
        
        # Generate random IP address
        ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
        
        # Calculate a risk score (0-100)
        risk_score = np.random.normal(30, 15)  # Most nodes have low-medium risk
        risk_score = max(0, min(100, risk_score))  # Clamp between 0-100
        
        nodes.append({
            'id': i,
            'ip': ip,
            'type': node_type,
            'security_level': security_level,
            'risk_score': risk_score,
            'active': random.random() > 0.1  # 90% of nodes are active
        })
    
    # Create connections between nodes
    connections = []
    for _ in range(num_connections):
        source = random.randint(0, num_nodes-1)
        target = random.randint(0, num_nodes-1)
        while target == source:  # Ensure no self-connections
            target = random.randint(0, num_nodes-1)
            
        traffic_volume = max(1, int(np.random.exponential(50)))  # Skewed distribution
        connection_type = random.choice(['HTTP', 'HTTPS', 'SSH', 'FTP', 'SMTP', 'DNS', 'SQL'])
        
        # Some connections are suspicious
        is_suspicious = random.random() < 0.05  # 5% chance of suspicious
        
        connections.append({
            'source': source,
            'target': target,
            'traffic_volume': traffic_volume,
            'connection_type': connection_type,
            'is_suspicious': is_suspicious
        })
    
    # Generate network traffic data points over time (last 24 hours)
    now = datetime.now()
    timestamps = [(now - timedelta(hours=24-i)).strftime("%Y-%m-%d %H:00:00") for i in range(25)]
    
    # Generate traffic patterns with some randomness
    base_traffic = [1000 + 500 * np.sin(i * np.pi / 12) for i in range(25)]  # Daily pattern
    traffic_data = [max(0, val + np.random.normal(0, val * 0.1)) for val in base_traffic]
    
    # Generate some attack attempts in the data
    attack_timestamps = []
    attack_types = []
    attack_sources = []
    attack_targets = []
    attack_severities = []
    
    num_attacks = random.randint(3, 8)
    for _ in range(num_attacks):
        # Random timestamp in the last 24 hours
        attack_time = now - timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        attack_timestamps.append(attack_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        attack_types.append(random.choice([
            'SQL Injection', 'XSS', 'DDoS', 'Brute Force', 'Man-in-the-Middle',
            'Phishing', 'Zero-day', 'Ransomware', 'Quantum Key Compromise'
        ]))
        
        attack_sources.append(f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}")
        attack_targets.append(random.choice([node['ip'] for node in nodes]))
        attack_severities.append(random.choice(['Low', 'Medium', 'High', 'Critical']))
    
    # Compile all data
    return {
        'nodes': nodes,
        'connections': connections,
        'traffic_data': {
            'timestamps': timestamps,
            'values': traffic_data
        },
        'attack_data': {
            'timestamps': attack_timestamps,
            'types': attack_types,
            'sources': attack_sources,
            'targets': attack_targets,
            'severities': attack_severities
        }
    }

def generate_threat_data(num_threats=5):
    """
    Generate simulated threat data for the dashboard.
    
    Args:
        num_threats: Number of threats to generate
    
    Returns:
        DataFrame with threat information
    """
    threat_types = [
        'SQL Injection', 'Cross-Site Scripting', 'DDoS Attack', 'Brute Force Login',
        'Man-in-the-Middle', 'Phishing Attempt', 'Zero-day Exploit', 'Ransomware',
        'Data Exfiltration', 'Privilege Escalation', 'Quantum Key Distribution Attack'
    ]
    
    sources = [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(num_threats)]
    targets = [f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}" for _ in range(num_threats)]
    types = [random.choice(threat_types) for _ in range(num_threats)]
    severities = [random.choice(['Low', 'Medium', 'High', 'Critical']) for _ in range(num_threats)]
    
    # Generate timestamps within the last 24 hours
    now = datetime.now()
    timestamps = [(now - timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )).strftime("%Y-%m-%d %H:%M:%S") for _ in range(num_threats)]
    
    # Probabilities of mitigation
    mitigation_status = [random.choice(['Mitigated', 'In Progress', 'Detected', 'Failed']) for _ in range(num_threats)]
    
    # Create the threat data DataFrame
    threat_data = pd.DataFrame({
        'Timestamp': timestamps,
        'Source': sources,
        'Target': targets,
        'Type': types,
        'Severity': severities,
        'Status': mitigation_status
    })
    
    # Sort by timestamp to have the most recent threats at the top
    threat_data = threat_data.sort_values('Timestamp', ascending=False).reset_index(drop=True)
    
    return threat_data
