"""Phoneme normalisation and the accent profile (which sound differences are tolerated)."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import yaml

VOWELS = set("aeiouɑɒɔæɛɪʊʌəɐɜɨøœyɯɵɤ")
STOPS = set("ptkbdɡgʈɖcɟqʔ")
FRICATIVES = set("fvθðszʃʒhçxɣβɸʂʐɕʑʝχħʕ")
NASALS = set("mnŋɲɳɴ")
LIQUIDS = set("lɫɭɹrɾɽɻjwʋʎ")
_STRIP = set("ːˑ:ʰʲˈˌ.ˤ^[]ʼ")
_ASPIRATED = {"th": "t", "kh": "k", "ph": "p"}


def broad_class(sym: str) -> str:
    if sym in VOWELS:
        return "vowel"
    if sym in STOPS:
        return "stop"
    if sym in FRICATIVES:
        return "fricative"
    if sym in NASALS:
        return "nasal"
    if sym in LIQUIDS:
        return "liquid"
    return "other"


class Unit(NamedTuple):
    symbol: str
    role: str | None = None   # None | "glide" (2nd part of a diphthong) | "rcolor" (post-vocalic r)


@dataclass(frozen=True)
class AccentProfile:
    name: str
    espeak_language: str
    model: str
    per_full_at: float
    per_zero_at: float
    word_present_min_coverage: float
    word_ok_above: float
    word_bad_below: float
    word_error_weight: float
    costs: dict[str, float]
    equivalent_groups: list[frozenset[str]]
    accepted: dict[str, frozenset[str]]
    near: frozenset[frozenset[str]]
    diphthongs: tuple[str, ...]
    affricates: tuple[str, ...]
    _group_of: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def sub_cost(self, expected: str, heard: str) -> float:
        if expected == heard or heard in self.accepted.get(expected, ()):
            return 0.0
        g = self._group_of.get(expected)
        if g is not None and self._group_of.get(heard) == g:
            return 0.0
        if frozenset((expected, heard)) in self.near:
            return self.costs["near_substitution"]
        if heard == "ə" and expected in VOWELS:
            return self.costs["reduction"]          # vowel reduction is normal in English
        ce, ch = broad_class(expected), broad_class(heard)
        if ce == ch and ce != "other":
            return self.costs["class_substitution"]
        return self.costs["substitution"]

    def deletion_cost(self, unit: Unit) -> float:
        if unit.role == "rcolor":
            return self.costs["rcolor_deletion"]
        if unit.role == "glide" or unit.symbol == "ə":
            return self.costs["cheap_deletion"]
        return self.costs["deletion"]

    def insertion_cost(self, symbol: str) -> float:
        return self.costs["cheap_insertion"] if symbol == "ə" else self.costs["insertion"]


def load_profile(path: str | Path) -> AccentProfile:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    groups = [frozenset(g) for g in data.get("equivalent", [])]
    group_of = {sym: i for i, g in enumerate(groups) for sym in g}
    costs = {"substitution": 1.0, "class_substitution": 0.6, "near_substitution": 0.3,
             "deletion": 1.0, "insertion": 1.0, "cheap_deletion": 0.4, "cheap_insertion": 0.4,
             "rcolor_deletion": 0.15, "reduction": 0.3}
    costs.update(data.get("costs", {}))
    th = data.get("thresholds", {})
    return AccentProfile(
        name=data["name"], espeak_language=data.get("espeak_language", "en-us"),
        model=data["model"], per_full_at=float(th.get("per_full_at", 0.12)),
        per_zero_at=float(th.get("per_zero_at", 0.60)),
        word_present_min_coverage=float(data.get("word_present_min_coverage", 0.5)),
        word_ok_above=float(th.get("word_ok_above", 0.9)),
        word_bad_below=float(th.get("word_bad_below", 0.6)),
        word_error_weight=float(th.get("word_error_weight", 0.5)),
        costs=costs, equivalent_groups=groups,
        accepted={k: frozenset(v) for k, v in (data.get("accepted") or {}).items()},
        near=frozenset(frozenset(p) for p in data.get("near", [])),
        diphthongs=tuple(data.get("diphthongs", [])), affricates=tuple(data.get("affricates", [])),
        _group_of=group_of)


def _clean(token: str) -> str:
    token = unicodedata.normalize("NFD", token)
    token = "".join(c for c in token if unicodedata.category(c) != "Mn")   # drop diacritics
    if token in _ASPIRATED:
        token = _ASPIRATED[token]
    token = "".join(c for c in token if c not in _STRIP and not c.isdigit())
    return token.replace("ᵻ", "ɪ").replace("ɚ", "əɹ")


def split_token(token: str, profile: AccentProfile) -> list[Unit]:
    """Split one model/espeak token into comparable units (see config/accents/indian_english.yaml)."""
    s = _clean(token)
    pairs = set(profile.diphthongs) | set(profile.affricates)
    units: list[Unit] = []
    i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two in pairs:
            if two in profile.affricates:
                units += [Unit(two[0]), Unit(two[1])]
            else:
                units += [Unit(two[0]), Unit(two[1], "glide")]
            i += 2
            continue
        ch = s[i]
        if ch == "ɹ" and units and units[-1].symbol in VOWELS and i > 0:
            units.append(Unit("ɹ", "rcolor"))
        else:
            units.append(Unit(ch))
        i += 1
    return units
