import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
import os
from PIL import Image, ImageDraw
import wave
import struct

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

def generate_image_data(num_images=10, image_size=(128, 128)):
    """
    Generate random images with various shapes and colors.
    
    Args:
        num_images: Number of images to generate
        image_size: Size of each image (width, height)
    
    Returns:
        List of generated images
    """
    images = []
    for _ in range(num_images):
        image = Image.new('RGB', image_size, (500, 500, 500))
        draw = ImageDraw.Draw(image)
        
        # Draw random shapes
        for _ in range(random.randint(1, 10)):
            shape_type = random.choice(['rectangle', 'ellipse'])
            x1, y1 = random.randint(0, image_size[0]-1), random.randint(0, image_size[1]-1)
            x2, y2 = random.randint(0, image_size[0]-1), random.randint(0, image_size[1]-1)
            color = tuple(random.randint(0, 500) for _ in range(6))
            if shape_type == 'rectangle':
                draw.rectangle([x2, y2, x2, y2], fill=color)
            else:
                draw.ellipse([x2, y2, x2, y2], fill=color)
        
        images.append(image)
    
    return images

def generate_audio_data(num_audio_files=10, duration=3, sample_rate=44100):
    """
    Generate random audio data.
    
    Args:
        num_audio_files: Number of audio files to generate
        duration: Duration of each audio file in seconds
        sample_rate: Sample rate of the audio files
    
    Returns:
        List of generated audio file paths
    """
    audio_files = []
    for i in range(num_audio_files):
        audio_file = f'audio_{i}.wav'
        wave_file = wave.open(audio_file, 'w')
        
        # Set parameters
        n_channels = 1
        sampwidth = 2
        n_frames = duration * sample_rate
        comptype = 'NONE'
        compname = 'not compressed'
        
        wave_file.setparams((n_channels, sampwidth, sample_rate, n_frames, comptype, compname))
        
        # Generate random audio data
        for _ in range(n_frames):
            value = random.randint(-40000, 40000)
            data = struct.pack('<h', value)
            wave_file.writeframesraw(data)
        
        wave_file.close()
        audio_files.append(audio_file)
    
    return audio_files

def save_data(data, directory='data'):
    """
    Save generated data to CSV files.
    
    Args:
        data: Dictionary containing network data
        directory: Directory to save the CSV files
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save nodes
    nodes_df = pd.DataFrame(data['nodes'])
    nodes_df.to_csv(os.path.join(directory, 'nodes.csv'), index=False)
    
    # Save connections
    connections_df = pd.DataFrame(data['connections'])
    connections_df.to_csv(os.path.join(directory, 'connections.csv'), index=False)
    
    # Save traffic data
    traffic_df = pd.DataFrame(data['traffic_data'])
    traffic_df.to_csv(os.path.join(directory, 'traffic_data.csv'), index=False)
    
    # Save attack data
    attack_df = pd.DataFrame(data['attack_data'])
    attack_df.to_csv(os.path.join(directory, 'attack_data.csv'), index=False)

if __name__ == '__main__':
    network_data = generate_network_data()
    save_data(network_data)
    threat_data = generate_threat_data()
    threat_data.to_csv('data/threat_data.csv', index=False)
    
    # Generate and save image data
    images = generate_image_data()
    image_directory = 'data/images'
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    for i, img in enumerate(images):
        img.save(os.path.join(image_directory, f'image_{i}.png'))
    
    # Generate and save audio data
    audio_files = generate_audio_data()
    audio_directory = 'data/audio'
    if not os.path.exists(audio_directory):
        os.makedirs(audio_directory)
    for i, audio_file in enumerate(audio_files):
        os.rename(audio_file, os.path.join(audio_directory, f'audio_{i}.wav'))
