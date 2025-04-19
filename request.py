import requests


response = requests.delete("http://localhost:8000/inventory/1")
print(response.json())

# Delete by name
response = requests.delete("http://localhost:8000/inventory/by-name/Aspirin")
print(response.json())