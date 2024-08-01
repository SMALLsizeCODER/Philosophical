import requests
import numpy as np

BASE_URL = "https://philosophical.onrender.com"

def post_request(endpoint, data):
    response = requests.post(f"{BASE_URL}/{endpoint}", json=data)
    return response.json()

def perceive(image):
    return post_request('perceive', {'image': image})

def remember(entity1, entity2, relation):
    return post_request('remember', {'entity1': entity1, 'entity2': entity2, 'relation': relation})

def think(context):
    return post_request('think', {'context': context})

def learn(episodes):
    return post_request('learn', {'episodes': episodes})

def chat(user_input):
    return post_request('chat', {'user_input': user_input})

def ping():
    return requests.get(f"{BASE_URL}/ping").json()

# Example usage
if __name__ == "__main__":
    # Example data
    image_data = np.random.rand(64, 64, 3).tolist()  # Replace with actual image data
    
    # Perceive
    print("Perception Output:", perceive(image_data))
    
    # Remember
    print("Remember Status:", remember("Python", "Programming", "is a type of"))
    
    # Think
    print("Reasoning Output:", think("Python"))
    
    # Learn
    print("Learning Status:", learn(100))
    
    # Chat
    print("AI Response:", chat("Tell me about Python."))
    
    # Ping
    print("Ping Status:", ping())
