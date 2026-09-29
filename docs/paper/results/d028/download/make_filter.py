"""Title regex for yt-dlp --match-filter: excludes MTG-QBH and MLEnd songs (E2b)."""

import json
import re

OVERRIDES = {
    "Bohemian Rapsody": r"bohemian\W+rhap?sody",
    "When I am sixty four": r"sixty\W*four",
    "You have got a friend": r"got\W+a\W+friend",
    "Samba de uma nota so": r"samba\W+de\W+uma\W+nota",
    "Agua de beber": r"gua\W+de\W+beber",
    "Lestaca": r"l\W?estaca",
    "Mediterraneo": r"mediterr.neo",
    "Cuando los angeles lloran": r"cuando\W+los\W+.ngeles",
    "Vivir asi es morir de amor": r"vivir\W+as.\W+es\W+morir",
    "Singin' in the Rain": r"singin\W*g?\W+in\W+the\W+rain",
    "Octopuss garden": r"octopus\W?s\W+garden",
    "Hedwig's Theme": r"hedwig",
    "Ob la di ob la da": r"ob\W*la\W*di",
    "Chega de saudade": r"chega\W+de\W+saudade",
}
GENERIC = {"help", "girl", "one", "wave", "america"}


def word_pattern(word: str) -> str:
    return r"\W?".join(re.escape(c) for c in word)


def title_pattern(artist: str, title: str) -> str:
    if title in OVERRIDES:
        return OVERRIDES[title]
    words = re.findall(r"[a-z0-9]+", title.lower())
    body = r"\W+".join(word_pattern(w) for w in words)
    if title.lower() in GENERIC:
        key = re.findall(r"[a-z0-9]+", artist.lower())[-1]
        return rf"(?=.*{key})(?=.*\b{body}\b)"
    return rf"\b{body}\b"


pairs = json.load(open("excl_titles.json"))
regex = "(?i)(?:" + "|".join(title_pattern(a, t) for a, t in pairs) + ")"
open("title_regex.txt", "w").write(regex)
print(len(regex))
