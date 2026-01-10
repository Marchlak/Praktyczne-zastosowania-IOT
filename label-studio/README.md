# Setup do label studio

Label studio to program do tworzenia labeli ;p

## UV

```bash
# synchronizujemy pakiety zadeklarowane w pyproject.toml
uv sync
```

Wszystko powinno pójść ok i na tym etapie możemy uruchomić `label studio` warto pamiętać o fladze, która umożliwi korzystanie z lokalnego systemu plików (importowanie plików przez UI działa OK tylko przy jednym bądź kilku plików).

```bash
LOCAL_FILES_SERVING_ENABLED=true uv run label-studio
```

Odwiedzamy localhost:8080 i wszystko powinno bzikać ;)
