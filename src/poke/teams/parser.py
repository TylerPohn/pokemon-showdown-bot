"""Parser for Showdown paste format."""
import re
from typing import Optional
from .models import Pokemon, Team

class TeamParser:
    """Parse Showdown paste format into Team objects."""

    # Regex patterns
    NAME_PATTERN = re.compile(
        r"^(?:(.+?)\s*\(([^)]+)\)|([^(@]+))(?:\s*\(([MF])\))?(?:\s*@\s*(.+))?$"
    )
    EV_PATTERN = re.compile(r"(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)", re.IGNORECASE)
    IV_PATTERN = re.compile(r"(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)", re.IGNORECASE)

    def parse_pokemon(self, text: str) -> Optional[Pokemon]:
        """Parse a single Pokemon from paste format."""
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return None

        pokemon = Pokemon(species="Unknown")

        # Parse first line (name/species/item)
        first_line = lines[0]
        if match := self.NAME_PATTERN.match(first_line):
            nickname, species1, species2, gender, item = match.groups()
            species = (species1 or species2 or "Unknown").strip()
            pokemon.species = species
            pokemon.nickname = nickname if nickname else None
            pokemon.gender = gender
            pokemon.item = item

        # Parse remaining lines
        for line in lines[1:]:
            line_lower = line.lower()

            if line_lower.startswith("ability:"):
                pokemon.ability = line.split(":", 1)[1].strip()

            elif line_lower.startswith("level:"):
                try:
                    pokemon.level = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line_lower.startswith("evs:"):
                ev_part = line.split(":", 1)[1]
                for match in self.EV_PATTERN.finditer(ev_part):
                    pokemon.evs[match.group(2)] = int(match.group(1))

            elif line_lower.startswith("ivs:"):
                iv_part = line.split(":", 1)[1]
                for match in self.IV_PATTERN.finditer(iv_part):
                    pokemon.ivs[match.group(2)] = int(match.group(1))

            elif "nature" in line_lower:
                pokemon.nature = line.split()[0]

            elif line_lower.startswith("tera type:"):
                pokemon.tera_type = line.split(":", 1)[1].strip()

            elif line_lower.startswith("shiny:"):
                pokemon.shiny = "yes" in line_lower

            elif line.startswith("-"):
                move = line[1:].strip()
                if move:
                    pokemon.moves.append(move)

        return pokemon

    def parse_team(self, text: str, team_id: Optional[str] = None, name: str = "Unnamed") -> Team:
        """Parse a full team from paste format."""
        # Split by double newline to get individual Pokemon
        pokemon_texts = re.split(r"\n\s*\n", text.strip())

        pokemon_list = []
        for ptext in pokemon_texts:
            if ptext.strip():
                mon = self.parse_pokemon(ptext)
                if mon:
                    pokemon_list.append(mon)

        if team_id is None:
            team_id = Team.generate_id(text)

        return Team(
            team_id=team_id,
            name=name,
            pokemon=pokemon_list,
        )
