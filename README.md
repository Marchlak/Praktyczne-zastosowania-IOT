# Praktyczne-zastosowania-IOT

## Info do wagi analogowej

Zdjęcia znajdują się w `WagaAnalogowaDataSet`.

Opisy są zdefiniowane w pliku `waga-analogowa-labele.json` 

```json}
    ],
    "drafts": [],
    "predictions": [],
    "data": {
      "image": "/data/local-files/?d=Users/rich/Downloads/WagaAnalogowaDataSet/IMG_6642.png"
    },
    "meta": {},
    "created_at": "2026-01-09T21:27:50.735908Z",
```

Opisy są zdefiniowane dla mojego eksportu więc ten plik można albo:
- potraktować jako przykład i wygenerować własny
- zamienić ścieżki regexem albo poprzez jakiś skrypt wygenerowany przez LLMa jak np. ten poniżej (nie daje gwarancji jego poprawnego działania)

```python
import json
import os
import math

# Ścieżka do Twojego wyeksportowanego pliku z Label Studio
JSON_PATH = 'project-1-at-2026-01-09.json'
# Ścieżka, gdzie faktycznie masz zdjęcia
IMAGE_DIR = '/Users/rich/Downloads/WagaAnalogowaDataSet/'

def calculate_weight(points, max_capacity=5.0):
    # points to słownik {label: (x, y)}
    if not all(k in points for k in ['center', 'tip', 'zero']):
        return None

    c = points['center']
    t = points['tip']
    z = points['zero']

    # Obliczamy kąty wektorów względem osi X
    angle_zero = math.atan2(z[1] - c[1], z[0] - c[0])
    angle_tip = math.atan2(t[1] - c[1], t[0] - c[0])

    # Różnica kątów (w radianach)
    diff = angle_tip - angle_zero
    
    # Normalizacja do zakresu 0 - 2PI (zgodnie z ruchem wskazówek)
    if diff < 0:
        diff += 2 * math.pi
        
    # Przeliczenie na kilogramy (zakładając, że 360 stopni = max_capacity)
    weight = (diff / (2 * math.pi)) * max_capacity
    return round(weight, 2)

# Wczytywanie danych
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

for entry in data:
    # Wyciągamy samą nazwę pliku ze ścieżki "/data/local-files/?d=..."
    original_path = entry['data']['image']
    file_name = os.path.basename(original_path.split('?d=')[-1])
    full_path = os.path.join(IMAGE_DIR, file_name)

    points = {}
    # Wyciąganie punktów z adnotacji
    for result in entry['annotations'][0]['result']:
        if result['type'] == 'keypointlabels':
            label = result['value']['keypointlabels'][0]
            # Współrzędne są w % wielkości obrazu (0-100)
            x = result['value']['x']
            y = result['value']['y']
            points[label] = (x, y)

    weight = calculate_weight(points)
    print(f"Plik: {file_name} -> Wykryta waga: {weight} kg")
```

### Dlaczego JSON?

Normalny format yolo eksportuje tylko dla targetów.

## Targety

Mamy 1 target:

- Tarcza

I 3 pinpointy:

- Zero (0 na wadze)
- Center (początek wskazówki)
- Tip (koniec wskazówki)

![Labele](labele.png)
