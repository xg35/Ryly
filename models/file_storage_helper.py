# my_project/models/file_storage_help.py


import os
import json

class FileStorageHelper:
    @staticmethod
    def load_lines(filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    @staticmethod
    def write_lines(filepath, lines):
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    @staticmethod
    def load_json(filepath):
        """Loads JSON from the given file, or returns an empty dict if file doesn't exist or is invalid."""
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def write_json(filepath, data):
        """Writes 'data' as JSON to the given file path."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)