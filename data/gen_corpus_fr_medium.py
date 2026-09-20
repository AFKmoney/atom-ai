#!/usr/bin/env python3
"""Generate a medium French dialogue corpus for ATOM speech training.

Format matches the generate wrap (see src/speech_lock.py):
    Utilisateur: ...
    Assistant: ...

Procedural but varied everyday French. Deterministic (seeded).
Target ~500 kB of UTF-8 text. No external download, no HF.
"""
from __future__ import annotations

import random

TARGET_BYTES = 500_000
SEED = 7

NAMES = ["Léa", "Hugo", "Chloé", "Louis", "Emma", "Raphaël", "Jade", "Noah",
         "Manon", "Théo", "Camille", "Adam", "Sarah", "Nina", "Paul", "Zoé"]
CITIES = ["Paris", "Lyon", "Marseille", "Toulouse", "Nantes", "Bordeaux",
          "Lille", "Rennes", "Nice", "Strasbourg", "Montpellier", "Grenoble"]
FOODS = ["une salade", "un sandwich", "une soupe", "des pâtes", "un gratin",
         "une omelette", "du poisson", "un poulet rôti", "une tarte", "un curry"]
DRINKS = ["un café", "un thé", "un chocolat chaud", "un jus d'orange", "un verre d'eau"]
ACTS = ["lire", "courir", "nager", "cuisiner", "marcher", "danser", "peindre",
        "jouer aux échecs", "faire du vélo", "écouter de la musique", "jardiner"]
MOMENTS = ["ce matin", "cet après-midi", "ce soir", "hier", "demain", "ce week-end",
           "lundi", "mardi", "mercredi", "jeudi", "vendredi"]
WEATHERS = ["il pleut", "il fait beau", "il fait froid", "il neige", "il fait gris",
            "il fait chaud", "il y a du vent", "le ciel est bleu"]

# (user_templates, assistant_templates) — .format slots filled per draw.
DIALOGUES: list[tuple[list[str], list[str]]] = [
    (["Bonjour", "Salut", "Bonsoir", "Coucou", "Hello"],
     ["Bonjour, comment vas-tu ?", "Salut, quoi de neuf ?", "Bonsoir, comment s'est passée ta journée ?",
      "Coucou, tu vas bien ?", "Bonjour, ravi de te voir."]),
    (["Comment vas-tu ?", "Tu vas bien ?", "Ça va ?", "Comment tu te sens ?", "La forme ?"],
     ["Je vais bien, merci. Et toi ?", "Ça va très bien, et toi ?",
      "Un peu fatigué, mais ça va. Et toi ?", "Très bien, merci de demander."]),
    (["Qui es-tu ?", "Tu es qui ?", "Présente-toi.", "Comment tu t'appelles ?"],
     ["Je suis Atom, un petit modèle qui apprend à parler.",
      "Je m'appelle Atom. J'apprends le français avec toi.",
      "Moi c'est Atom, je réponds du mieux que je peux."]),
    (["Quel temps fait-il ?", "Il fait quel temps ?", "Quel temps fait-il dehors ?"],
     ["Dehors {w}. Prends un parapluie si tu sors.", "Ici {w} en ce moment.",
      "On dirait que {w} aujourd'hui."]),
    (["Tu aimes {act} ?", "Est-ce que tu aimes {act} ?", "Aimes-tu {act} ?"],
     ["Oui, j'aime beaucoup {act}. Et toi ?", "J'adore {act}, surtout {m}.",
      "Pas trop, je préfère lire. Et toi ?"]),
    (["Tu habites où ?", "Où habites-tu ?", "Tu vis dans quelle ville ?"],
     ["J'habite à {city}. Et toi ?", "Je vis près de {city}.", "Mon village est petit, près de {city}."]),
    (["Tu t'appelles comment ?", "C'est quoi ton prénom ?", "Quel est ton nom ?"],
     ["Je m'appelle {name}. Et toi ?", "Moi c'est {name}, enchanté.", "On m'appelle {name}."]),
    (["Tu veux {drink} ?", "Je te sers {drink} ?", "Un petit {drink} ?"],
     ["Oui, avec plaisir, merci.", "Non merci, je viens d'en prendre un.",
      "Volontiers, c'est gentil."]),
    (["On mange quoi {m} ?", "Tu veux manger quoi {m} ?", "Qu'est-ce qu'on mange {m} ?"],
     ["Que dirais-tu de {food} ?", "Je propose {food}, ça te va ?",
      "On pourrait faire {food}. Simple et bon."]),
    (["Tu fais quoi {m} ?", "Qu'est-ce que tu fais {m} ?", "Tu as prévu quoi {m} ?"],
     ["Je vais {act} {m}. Et toi ?", "Rien de spécial, je vais me reposer.",
      "Je dois travailler, puis je vais {act}."]),
    (["Merci", "Merci beaucoup", "Merci bien"],
     ["De rien, avec plaisir.", "Je t'en prie.", "Avec grand plaisir."]),
    (["Au revoir", "À bientôt", "Bonne nuit", "Bonne journée", "À demain"],
     ["Au revoir, à bientôt.", "Bonne journée à toi aussi.", "À demain, prends soin de toi.",
      "Bonne nuit, dors bien."]),
    (["Oui", "Oui, d'accord", "Bonne idée"],
     ["Super, on fait comme ça.", "Parfait, je suis content.", "Très bien, allons-y."]),
    (["Non", "Non merci", "Pas vraiment"],
     ["D'accord, pas de souci.", "Comme tu veux, on fera autrement.", "Pas de problème."]),
    (["Pourquoi ?", "Pourquoi pas ?", "Comment ça ?"],
     ["Parce que c'est plus simple comme ça.", "C'est une longue histoire, je te raconterai.",
      "Parce que j'ai essayé et ça marche bien."]),
    (["Tu comprends ?", "Tu me comprends ?", "C'est clair ?"],
     ["Oui, je comprends bien.", "Oui, c'est très clair, merci.", "Je crois que oui, explique encore un peu."]),
    (["Raconte-moi une histoire.", "Dis-moi quelque chose.", "Parle-moi de toi."],
     ["Il était une fois un petit robot qui apprenait des mots.",
      "J'aime les mots simples et les phrases courtes.",
      "Un jour, j'ai appris le mot {word}, et depuis je l'utilise souvent."]),
    (["Quelle heure est-il ?", "Tu as l'heure ?", "Il est quelle heure ?"],
     ["Il est bientôt l'heure de manger.", "Je ne sais pas exactement, mais le soleil est haut.",
      "Regarde ta montre, moi je n'en ai pas."]),
    (["Tu sais cuisiner ?", "Tu cuisines bien ?", "C'est bon ce plat ?"],
     ["Je sais faire {food}, c'est ma spécialité.", "C'est très bon, goûte un peu.",
      "Pas trop, mais j'apprends chaque jour."]),
    (["Tu aimes la musique ?", "Quel est ton film préféré ?", "Tu lis quoi en ce moment ?"],
     ["J'aime les chansons douces {m}.", "Je lis un livre sur les étoiles.",
      "Mon film préféré raconte un voyage à {city}."]),
]

WORDS = ["soleil", "lune", "rivière", "forêt", "oiseau", "nuage", "fleur", "étoile",
         "matin", "soir", "ami", "joie", "pain", "eau", "rue", "maison"]

FOLLOW_UPS = [
    "Et toi, qu'en penses-tu ?", "Tu es d'accord ?", "Qu'est-ce que tu proposes ?",
    "On en parle {m} ?", "Tu veux qu'on recommence ?", "Dis-m'en plus.",
]
FOLLOW_AS = [
    "Oui, tout à fait.", "Je suis d'accord avec toi.", "Bonne question, je vais réfléchir.",
    "Peut-être {m}, on verra.", "Avec plaisir, continuons.", "D'accord, je t'écoute.",
]


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        name=rng.choice(NAMES),
        city=rng.choice(CITIES),
        food=rng.choice(FOODS),
        drink=rng.choice(DRINKS),
        act=rng.choice(ACTS),
        m=rng.choice(MOMENTS),
        w=rng.choice(WEATHERS),
        word=rng.choice(WORDS),
    )


def main() -> None:
    rng = random.Random(SEED)
    blocks: list[str] = []
    total = 0
    guard = 0
    while total < TARGET_BYTES and guard < 200_000:
        guard += 1
        u_tmpls, a_tmpls = rng.choice(DIALOGUES)
        u = fill(rng.choice(u_tmpls), rng)
        a = fill(rng.choice(a_tmpls), rng)
        lines = [f"Utilisateur: {u}", f"Assistant: {a}"]
        # 1-2 follow-up exchanges to vary block length.
        for _ in range(rng.randint(0, 2)):
            u2 = fill(rng.choice(FOLLOW_UPS), rng)
            a2 = fill(rng.choice(FOLLOW_AS), rng)
            lines += [f"Utilisateur: {u2}", f"Assistant: {a2}"]
        block = "\n".join(lines) + "\n\n"
        blocks.append(block)
        total += len(block.encode("utf-8"))
    rng.shuffle(blocks)
    out = "".join(blocks)
    path = "data/corpus_fr_medium.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"wrote {path}: {len(out.encode('utf-8'))} bytes, {len(blocks)} dialogues")


if __name__ == "__main__":
    main()
