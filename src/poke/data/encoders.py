"""Encoders for converting Pokemon data to numeric IDs."""
import json
from pathlib import Path
from typing import Dict, Optional

class SpeciesEncoder:
    """Encode Pokemon species names to numeric IDs."""

    def __init__(self, species_list: Optional[list[str]] = None):
        self._species_to_id: Dict[str, int] = {"<UNK>": 0}
        self._id_to_species: Dict[int, str] = {0: "<UNK>"}

        if species_list:
            for species in species_list:
                self.add(species)

    def add(self, species: str) -> int:
        """Add a species and return its ID."""
        normalized = self._normalize(species)
        if normalized not in self._species_to_id:
            new_id = len(self._species_to_id)
            self._species_to_id[normalized] = new_id
            self._id_to_species[new_id] = normalized
        return self._species_to_id[normalized]

    def encode(self, species: str) -> int:
        """Encode a species name to ID."""
        normalized = self._normalize(species)
        return self._species_to_id.get(normalized, 0)

    def decode(self, id: int) -> str:
        """Decode an ID to species name."""
        return self._id_to_species.get(id, "<UNK>")

    def _normalize(self, species: str) -> str:
        """Normalize species name."""
        return species.lower().replace(" ", "").replace("-", "")

    def save(self, path: Path) -> None:
        """Save encoder to JSON."""
        path.write_text(json.dumps(self._species_to_id))

    @classmethod
    def load(cls, path: Path) -> "SpeciesEncoder":
        """Load encoder from JSON."""
        encoder = cls()
        data = json.loads(path.read_text())
        encoder._species_to_id = data
        encoder._id_to_species = {v: k for k, v in data.items()}
        return encoder

    def __len__(self) -> int:
        return len(self._species_to_id)


class MoveEncoder:
    """Encode move names to numeric IDs."""

    def __init__(self):
        self._move_to_id: Dict[str, int] = {"<UNK>": 0}
        self._id_to_move: Dict[int, str] = {0: "<UNK>"}

    def add(self, move: str) -> int:
        normalized = move.lower().replace(" ", "")
        if normalized not in self._move_to_id:
            new_id = len(self._move_to_id)
            self._move_to_id[normalized] = new_id
            self._id_to_move[new_id] = normalized
        return self._move_to_id[normalized]

    def encode(self, move: str) -> int:
        normalized = move.lower().replace(" ", "")
        return self._move_to_id.get(normalized, 0)

    def decode(self, id: int) -> str:
        return self._id_to_move.get(id, "<UNK>")

    def __len__(self) -> int:
        return len(self._move_to_id)


class StatusEncoder:
    """Encode status conditions to numeric IDs."""
    STATUS_MAP = {
        "": 0,
        "brn": 1,
        "frz": 2,
        "par": 3,
        "psn": 4,
        "tox": 5,
        "slp": 6,
    }

    @classmethod
    def encode(cls, status: str) -> int:
        return cls.STATUS_MAP.get(status.lower(), 0)


class WeatherEncoder:
    """Encode weather conditions."""
    WEATHER_MAP = {
        "": 0,
        "none": 0,
        "sunnyday": 1,
        "raindance": 2,
        "sandstorm": 3,
        "hail": 4,
        "snow": 5,
    }

    @classmethod
    def encode(cls, weather: Optional[str]) -> int:
        if weather is None:
            return 0
        return cls.WEATHER_MAP.get(weather.lower(), 0)
