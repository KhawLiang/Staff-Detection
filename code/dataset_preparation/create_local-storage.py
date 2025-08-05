import json

l = []

for i in range(0, 1341, 5):
    l.append({'image': f"/data/local-files?d=img/frame_{i}.jpg"})

with open("local-storage.json", "w") as file:
    json.dump(l, file, indent=4)