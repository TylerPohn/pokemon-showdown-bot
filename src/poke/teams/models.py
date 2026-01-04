"""Data models for Pokemon teams."""
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib

@dataclass
class Pokemon:
    """A single Pokemon on a team."""
    species: str
    nickname: Optional[str] = None
    item: Optional[str] = None
    ability: Optional[str] = None
    evs: dict[str, int] = field(default_factory=dict)
    ivs: dict[str, int] = field(default_factory=dict)
    nature: Optional[str] = None
    moves: List[str] = field(default_factory=list)
    level: int = 100
    gender: Optional[str] = None
    shiny: bool = False
    tera_type: Optional[str] = None

    def to_showdown(self) -> str:
        """Convert to Showdown paste format."""
        lines = []

        # Name line
        name_line = self.species
        if self.nickname:
            name_line = f"{self.nickname} ({self.species})"
        if self.gender:
            name_line += f" ({self.gender})"
        if self.item:
            name_line += f" @ {self.item}"
        lines.append(name_line)

        if self.ability:
            lines.append(f"Ability: {self.ability}")

        if self.level != 100:
            lines.append(f"Level: {self.level}")

        if self.shiny:
            lines.append("Shiny: Yes")

        if self.tera_type:
            lines.append(f"Tera Type: {self.tera_type}")

        if self.evs:
            ev_str = " / ".join(f"{v} {k}" for k, v in self.evs.items() if v > 0)
            if ev_str:
                lines.append(f"EVs: {ev_str}")

        if self.ivs:
            iv_str = " / ".join(f"{v} {k}" for k, v in self.ivs.items() if v < 31)
            if iv_str:
                lines.append(f"IVs: {iv_str}")

        if self.nature:
            lines.append(f"{self.nature} Nature")

        for move in self.moves:
            lines.append(f"- {move}")

        return "\n".join(lines)

    def to_packed(self) -> str:
        """Convert to packed format for poke-env.

        Packed format: nickname|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|misc
        Where misc = happiness,pokeball,hiddenpowertype,gigantamax,dynamaxlevel,teratype
        """
        def to_id(s: str) -> str:
            """Convert to Showdown ID format (lowercase, no spaces/special chars)."""
            if not s:
                return ""
            return "".join(c.lower() for c in s if c.isalnum())

        # Format: nickname|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|misc
        parts = []

        # Nickname = species name (if no nickname, use species as nickname)
        parts.append(self.nickname or self.species)

        # Species (empty if same as nickname, which is our default)
        parts.append("")

        # Item
        parts.append(to_id(self.item) if self.item else "")

        # Ability
        parts.append(to_id(self.ability) if self.ability else "")

        # Moves (comma-separated)
        moves = ",".join(to_id(m) for m in self.moves)
        parts.append(moves)

        # Nature (capitalized)
        parts.append(self.nature if self.nature else "")

        # EVs (format: hp,atk,def,spa,spd,spe - empty for 0)
        ev_order = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
        evs = ",".join(str(self.evs.get(stat, 0)) if self.evs.get(stat, 0) > 0 else "" for stat in ev_order)
        parts.append(evs)

        # Gender
        parts.append(self.gender or "")

        # IVs (format: hp,atk,def,spa,spd,spe - empty for 31)
        ivs = ",".join(str(self.ivs.get(stat, 31)) if self.ivs.get(stat, 31) < 31 else "" for stat in ev_order)
        parts.append(ivs)

        # Shiny
        parts.append("S" if self.shiny else "")

        # Level
        parts.append(str(self.level) if self.level != 100 else "")

        # Misc field: happiness,pokeball,hiddenpowertype,gigantamax,dynamaxlevel,teratype
        misc_parts = [
            "",  # happiness (empty = default 255)
            "",  # pokeball
            "",  # hiddenpowertype
            "",  # gigantamax
            "",  # dynamaxlevel
            self.tera_type if self.tera_type else "",  # teratype (capitalized)
        ]
        parts.append(",".join(misc_parts))

        return "|".join(parts)


@dataclass
class Team:
    """A complete Pokemon team."""
    team_id: str
    name: str
    pokemon: List[Pokemon]
    format: str = "gen9ou"
    source: Optional[str] = None  # e.g., "smogon_samples"

    def __len__(self) -> int:
        return len(self.pokemon)

    def to_showdown(self) -> str:
        """Convert team to Showdown paste format."""
        return "\n\n".join(p.to_showdown() for p in self.pokemon)

    def to_packed(self) -> str:
        """Convert team to packed format for poke-env.

        Packed format: species|ability|item|moves|nature|evs|gender|ivs|shiny|level|happiness|pokeball|hiddenpowertype|gigantamax|dynamaxlevel|teratype|
        Separated by ]
        """
        packed_pokemon = []
        for mon in self.pokemon:
            packed_pokemon.append(mon.to_packed())
        return "]".join(packed_pokemon)

    @staticmethod
    def generate_id(content: str) -> str:
        """Generate stable team ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]
