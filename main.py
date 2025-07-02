from data import load_doc, clean_pairs, to_pairs, save_clean_data, load_clean_sentences, max_length
from tokenizer import create_tokenizer, encode_sequences, encode_output
from numpy.random import shuffle
from model import define_model, plot_model, load_model, evaluate_model, predict_sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from argparse import ArgumentParser
from time import time
from itertools import product

import inflect

from string import punctuation
from rich.console import Console
from rich.progress import Progress
from rich.table import Table


console = Console()


def convert_to_past_tense(word):
    irregular_verbs = {
        'say': 'said',
        'go': 'went',
        'do': 'did',
        'have': 'had',
        'be': 'been',
        'see': 'saw',
        'come': 'came',
        'run': 'ran',
        'drink': 'drank',
        'eat': 'ate',
        'write': 'wrote',
        'take': 'took',
        'give': 'gave',
        'find': 'found',
        'think': 'thought',
        'make': 'made',
        'know': 'knew',
        'begin': 'began',
        'fly': 'flew',
        'break': 'broke',
        'choose': 'chose',
        'drive': 'drove',
        'wear': 'wore',
        'swim': 'swam',
        'ring': 'rang',
        'run': 'ran',
        'sing': 'sang',
        'speak': 'spoke',
        'teach': 'taught',
        'catch': 'caught',
        'bring': 'brought',
        'think': 'thought',
        'fight': 'fought',
        'seek': 'sought',
        'leave': 'left',
        'meet': 'met',
        'read': 'read',
        'sleep': 'slept',
        'sell': 'sold',
        'tell': 'told',
        'stand': 'stood',
        'win': 'won',
        'lose': 'lost',
        'pay': 'paid',
        'send': 'sent',
        'build': 'built',
        "feel": "felt"
    }
    
    if word in irregular_verbs:
        return irregular_verbs[word]
    elif word.endswith('e'):
        return word + 'd'
    elif word.endswith(('y', 'ay', 'ey', 'iy', 'oy', 'uy')):
        return word + 'ed'
    elif word.endswith('le'):
        return word[:-2] + 'led'
    else:
        return word + 'ed'
    
def adjective_to_adverb(adj):
    irregular = {
        "good": "well",
        "fast": "fast",
        "hard": "hard",
        "late": "late",
        "early": "early"
    }

    if adj in irregular:
        return irregular[adj]

    if adj.endswith("y"):
        return adj[:-1] + "ily"     # happy → happily
    elif adj.endswith("le"):
        return adj[:-1] + "y"       # gentle → gently
    elif adj.endswith("ic"):
        return adj + "ally"         # basic → basically
    else:
        return adj + "ly"           # slow → slowly

def generate_data(args):
    p = inflect.engine()

    nouns = {
        "cav": ["house", "home"],
        "ken": ["language"],
        "hund": ["dog"],
        "wán": ["cat"],
        "palta": ["something"],
        "crum": ["food"],
        "crųs": ["fruit"],
        "valis": ["vegetable"],
        "Gut": ["God"],
        "dai": ["day"],
        "noon": ["night"],
        "sut": ["now"],
        "ţoc": ["later"],
        "döka": ["today"],
        "tųka": ["tomorrow"],
        "söka": ["yesterday"],
        "eset": ["it"],
        "vost": ["bird"],
        "dųnţ": ["idiot"],
        "ţweve": ["knee"],
        "rųnar": ["leg"],
        "tųs": ["foot"],
        "mųl": ["toe"],
        "malųs": ["elbow"],
        "tųsar": ["arm"],
        "lųsö": ["hand"],
        "cráská": ["finger"],
        "cröma": ["head"],
        "geseţ": ["face"],
        "hųdar": ["shoulder"],
        "hadint": ["skin"],
        "shwųs": ["menu"],
        "reţet": ["check", "bill"],
        "Keken": ["Keken"],
        "Enginiţ": ["English"],
        "Gorgas": ["Gorgus"],
        "hawai": ["friend"],
        "casţ": ["mixture"],
        "crųscasţ": ["fruit salad"],
        "valiscasţ": ["vegetable salad"],
        "gene": ["voice"],
        "kaiţen": ["case"],
        "kųsţöm": ["tool"],
        "kųnusi": ["creation"],
        "nasų": ["name"],
        "ţosok": ["apple"],
        "bröţ": ["bread"],
        "samat": ["king"],
        "omnis": ["everything"],
        "baţal": ["war"],
        "vie": ["hell"],
        "walas": ["heaven"],
        "sele": ["sky"],
        "malös": ["fish"],
        "vöcas": ["mouth"],
        "sira": ["eye"]
    }
    pronouns = {
        "kal": ["you", "yourself"],
        "mas": ["me", "myself"],
        "zali": ["you", "you guys", "you all"]
    }
    noun_prefixes = {
        "val": "the {noun}",
        "we": "{plural_noun}",
        "valwe": "the {noun}s",
        "zut": "no {noun}",
        "wezut": "no {noun}s",
        "pis": "your {noun}",
        "piswe": "your {noun}s",
        "piko": "my {noun}",
        "pikowe": "my {noun}s",
        "pala": "their {noun}",
        "palawe": "their {noun}s",
        "pili": "our {noun}",
        "piliwe": "our {noun}s",
    }
    verbs = {
        "ki": ["like"],
        "duwá": ["love"],
        "plaki": ["love"],
        "harl": ["hate", "dislike"],
        "ţwa": ["eat"],
        "las": ["drink"],
        "marko": ["float"],
        "tai": ["feel"],
        "s̄am": ["stink"],
        "snaile": ["smell"],
        "kųzö": ["see"],
        "stųp": ["stop"],
        "stai": ["start"],
        "kil": ["rejoice"],
        "ţes̄": ["use"],
        "as̄ai": ["create", "make"],
        "sai": ["see"],
        "sas̄a": ["ask"],
        "haiţe": ["bark"],
        "lopawe": ["look like"],
        "pia": ["become"],
        "spilse": ["play"],
        "s̄alse": ["play"],
        "mota": ["walk"],
        "speţi": ["run"],
        "möá": ["sleep", "rest"],
        "ţamak": ["hit", "beat"],
        "panali": ["move"],
        "saiţa": ["go"]
    }
    phrases = {
        "takţa": ["is", "be", "am"],
        "maiya": ["hi", "sup", "yo", "hey"],
        "hwádan": ["hello"],
        "wemaiya": ["greetings"],
        "taitamţa palta?": ["How are you?", "What's up?", "Do you feel something?"],
        "tara": ["thanks", "thank you"],
        "báká": ["bye", "goodbye", "see you later"],
        "zai": ["damn", "wow"],
        "ke": ["yes", "yeah"],
        "maiya, taitamţa palta?": ["hi, how are you?", "hey, how are you?", "sup, what's up?", "yo, what's up?"],
        "hwádan, taitamţa palta?": ["hello, how are you?"],
        "wemaiya, taitamţa palta?": ["greetings, how are you?"],

        "val": ["the"],
        "wa": ["a", "an"],
        "pis": ["your"],
        "piko": ["my"],
        "pala": ["their"],
        "pili": ["our"],
        "to": ["he"],
        "ta": ["she"],
        "ma": ["they"],
        "me": ["they"],
        "se": ["they"],
        "ko": ["I"],
        "tam": ["you"],
        "tas": ["you guys", "you all"],
        "zol": ["to"],
        "zut": ["no", "not"]
    }
    adjectives = {
        "ke": ["good"],
        "de": ["bad"],
        "ţųs̄": ["alright", "ok", "okay"],
        "kimö": ["happy"],
        "don": ["sad"],
        'graz': ["angry", 'mad'],
        "ja": ["smart"],
        "kinö": ["stupid", "dumb"],
        "tom": ["fun"],
        "salam": ["kind", "nice"],
        "yök": ["gross", "yucky", "messy"],
        "hugö": ["big", "large"],
        "tip": ["small", "tiny"],
        "duö": ["tall"],
        "sap": ["short"],
        "dös̄": ["thin"],
        "als̄": ["thick"],
        "zoo": ["fast", "quick"],
        "snai": ["slow"],
        "cöwi": ["cute"],
        "cömö": ["cool"],
        "hait": ["hot"],
        "ţili": ["cold"],
        "wol": ["new"],
        "ţös": ["old"],
        "ziut": ["rough", "scratchy"],
        "silö": ["smooth"],
        "s̄ais": ["soft"],
        "varda": ["annoying"],
        "sţwala": ["funny"],
        "hwede": ["silly"],
        "miţa": ["respectful"],
        "lawt": ["loud"],
        "kwis": ["quiet"],
        "wöja": ["excited"],
        "leţe": ["tasty"],
        "bas̄ad": ["beautiful"],
        "kas": ["hard to believe"]
    }
    sentence_templates = {
        #"{noun} takţa {adjective}": "{noun} is {adjective}",
        "{definite_noun} takţa {adjective}": "{definite_noun} is {adjective}",
        "takţa {definite_noun} {adjective}": "is {definite_noun} {adjective}?",
        "{plural_noun} takţa {adjective}": "{plural_noun} are {adjective}",
        "takţa {plural_noun} {adjective}": "are {plural_noun} {adjective}?",
        "{definite_plural_noun} takţa {adjective}": "{definite_plural_noun} are {adjective}",
        "takţa {definite_plural_noun} {adjective}": "are {definite_plural_noun} {adjective}?",
        "loktakţa {adjective}": "it is {adjective}",
        "meptakţa {adjective}": "that is {adjective}",
        "meptakţa {adjective}": "this is {adjective}",
        "totakţa {adjective}": "he is {adjective}",
        "tatakţa {adjective}": "she is {adjective}",
        "matakţa {adjective}": "they are {adjective}",
        "metakţa {adjective}": "they are {adjective}",
        "setakţa {adjective}": "they are {adjective}",
        "kotakţa {adjective}": "you are {adjective}",
        "tamtakţa {adjective}": "I am {adjective}",
        "tastakţa {adjective}": "you all are {adjective}",
        "tastakţa {adjective}": "you guys are {adjective}",

        "taklokţa {adjective}": "is it {adjective}?",
        "takmepţa {adjective}": "is that {adjective}?",
        "takmepţa {adjective}": "is this {adjective}?",
        "taktoţa {adjective}": "is he {adjective}?",
        "taktaţa {adjective}": "is she {adjective}?",
        "takmaţa {adjective}": "are they {adjective}?",
        "takmeţa {adjective}": "are they {adjective}?",
        "takseţa {adjective}": "are they {adjective}?",
        "takkoţa {adjective}": "are you {adjective}?",
        "taktamţa {adjective}": "am I {adjective}?",
        "taktasţa {adjective}": "are you all {adjective}?",
        "taktasţa {adjective}": "are you guys {adjective}?",
        #"{verb} {definite_noun} {adjective}": "{verb} the {adjective} {definite_noun}",
        #"{verb} {adverb} {plural_noun}": "{verb} {adverb} {plural_noun}",
        "{verb} {plural_noun}": "{verb} {plural_noun}",
        "{definite_noun} {verb}": "{definite_noun} {verb}s",
        #"{verb} {adverb} {definite_noun} {adjective}": "{verb} {adverb} the {adjective} {noun}",
        #"{verb} {adverb} {definite_noun} {adjective}": "{verb} the {adjective} {noun}",
        "{verb} {definite_plural_noun}": "{verb} {definite_plural_noun}",
        "{verb} {definite_plural_noun} {adjective}": "{verb} the {adjective} {plural_noun}",
        "{verb} {adverb}": "{verb} {adverb}",
    }

    possible_nouns = {}
    possible_adjectives = {}
    possible_verbs = {}
    adverbs = {}

    start = time()
    with Progress(console=console) as progress:
        task1 = progress.add_task("Generating basic phrases..", total=None)
        data = []

        console.print("[d]Handwritten phrases..")
        for keken, english in phrases.items():
            for eng in english:
                data.append(f"{keken}\t{eng}\n")
        
        console.print("[d]Pronouns..")
        for keken, english in pronouns.items():
            for eng in english:
                data.append(f"{keken}\t{eng}\n")
        
        console.print("[d]Nouns..")
        for keken, english in nouns.items():
            for eng in english:
                data.append(f"{keken}\t{eng}\n")
                possible_nouns[keken] = eng

                data.append(f"wa{keken}\t{p.a(eng)}\n")
                possible_nouns[f"wa{keken}"] = p.a(eng)

                for kek_pre, eng_pre in noun_prefixes.items():
                    data.append(f"{kek_pre}{keken}\t{eng_pre.replace('{noun}', eng).replace('{plural_noun}', p.plural(eng))}\n")
                    possible_nouns[f"{kek_pre}{keken}"] = eng_pre.replace("{noun}", eng).replace('{plural_noun}', p.plural(eng))
        
        console.print("[d]Adjectives..")
        for keken, english in adjectives.items():
            for eng in english:
                data.append(f"{keken}\t{eng}\n")
                possible_adjectives[keken] = [eng]

                data.append(f"{keken}zut\tnot {eng}\n")
                possible_adjectives[f"{keken}zut"] = [f"not {eng}"]

                data.append(f"{keken}sá\tvery {eng}\n")
                possible_adjectives[f"{keken}sá"] = [f"very {eng}"]
                data.append(f"{keken}sâ\tmore {eng}\n")
                possible_adjectives[f"{keken}sâ"] = [f"more {eng}"]
                data.append(f"{keken}sásâ\tmuch more {eng}\n")
                possible_adjectives[f"{keken}sásâ"] = [f"much more {eng}"]
                data.append(f"{keken}sàsâ\tsomewhat more {eng}\n")
                possible_adjectives[f"{keken}sàsâ"] = [f"somewhat more {eng}"]
                data.append(f"{keken}sà\tsomewhat {eng}\n")
                possible_adjectives[f"{keken}sà"] = [f"somewhat {eng}"]
                data.append(f"{keken}sǎ\tless {eng}\n")
                possible_adjectives[f"{keken}sǎ"] = [f"less {eng}"]
                data.append(f"{keken}sásǎ\tmuch less {eng}\n")
                possible_adjectives[f"{keken}sásǎ"] = [f"much less {eng}"]
                data.append(f"{keken}sàsǎ\tsomewhat less {eng}\n")
                possible_adjectives[f"{keken}sàsǎ"] = [f"somewhat less {eng}"]

                if keken[-1] in "aeiouo\u0308\u0304":
                    data.append(f"{keken}tē\t{adjective_to_adverb(eng)}\n")
                    adverbs[f"{keken}tē"] = [adjective_to_adverb(eng)]
                else:
                    data.append(f"{keken}ē\t{adjective_to_adverb(eng)}\n")
                    adverbs[f"{keken}ē"] = [adjective_to_adverb(eng)]
        
        console.print("[d]Verbs..")
        for keken, english in verbs.items():
            for eng in english:
                data.append(f"{keken}\t{eng}\n")
                #possible_verbs[f"{keken}"] = [f"{eng}"]

                # present tense
                data.append(f"lok{keken}ţa\tit {eng}s\n")
                possible_verbs[f"lok{keken}ţa"] = [f"it {eng}s"]
                data.append(f"mep{keken}ţa\tthis {eng}s\n")
                data.append(f"mep{keken}ţa\tthat {eng}s\n")
                possible_verbs[f"mep{keken}ţa"] = [f"this {eng}s", f"that {eng}s"]
                data.append(f"to{keken}ţa\the {eng}s\n")
                possible_verbs[f"to{keken}ţa"] = [f"he {eng}s"]
                data.append(f"ta{keken}ţa\tshe {eng}s\n")
                possible_verbs[f"ta{keken}ţa"] = [f"she {eng}s"]
                data.append(f"ma{keken}ţa\tthey {eng}\n")
                possible_verbs[f"ma{keken}ţa"] = [f"they {eng}"]
                data.append(f"me{keken}ţa\tthey {eng}\n")
                possible_verbs[f"me{keken}ţa"] = [f"they {eng}"]
                data.append(f"se{keken}ţa\tthey {eng}\n")
                possible_verbs[f"se{keken}ţa"] = [f"they {eng}"]
                data.append(f"ko{keken}ţa\tI {eng}\n")
                possible_verbs[f"ko{keken}ţa"] = [f"I {eng}"]
                data.append(f"tam{keken}ţa\tyou {eng}\n")
                possible_verbs[f"tam{keken}ţa"] = [f"you {eng}"]
                data.append(f"tas{keken}ţa\tyou guys {eng}\n")
                data.append(f"tas{keken}ţa\tyou all {eng}\n")
                possible_verbs[f"tas{keken}ţa"] = [f"you guys {eng}", f"you all {eng}"]
                data.append(f"zol{keken}ţa\tto {eng}\n")
                possible_verbs[f"zol{keken}"] = [f"to {eng}"]
                
                # past tense
                data.append(f"lok{keken}alt\tit {convert_to_past_tense(eng)}\n")
                possible_verbs[f"lok{keken}alt"] = [f"it {convert_to_past_tense(eng)}"]
                data.append(f"mep{keken}alt\tthis {convert_to_past_tense(eng)}\n")
                data.append(f"mep{keken}alt\tthat {convert_to_past_tense(eng)}\n")
                possible_verbs[f"mep{keken}alt"] = [f"this {convert_to_past_tense(eng)}", f"that {convert_to_past_tense(eng)}"]
                data.append(f"to{keken}alt\the {convert_to_past_tense(eng)}\n")
                possible_verbs[f"to{keken}alt"] = [f"he {convert_to_past_tense(eng)}"]
                data.append(f"ta{keken}alt\tshe {convert_to_past_tense(eng)}\n")
                possible_verbs[f"ta{keken}alt"] = [f"she {convert_to_past_tense(eng)}"]
                data.append(f"ma{keken}alt\tthey {convert_to_past_tense(eng)}\n")
                possible_verbs[f"ma{keken}alt"] = [f"they {convert_to_past_tense(eng)}"]
                data.append(f"me{keken}alt\tthey {convert_to_past_tense(eng)}\n")
                possible_verbs[f"me{keken}alt"] = [f"they {convert_to_past_tense(eng)}"]
                data.append(f"se{keken}alt\tthey {convert_to_past_tense(eng)}\n")
                possible_verbs[f"se{keken}alt"] = [f"they {convert_to_past_tense(eng)}"]
                data.append(f"ko{keken}alt\tI {convert_to_past_tense(eng)}\n")
                possible_verbs[f"ko{keken}alt"] = [f"I {convert_to_past_tense(eng)}"]
                data.append(f"tam{keken}alt\tyou {convert_to_past_tense(eng)}\n")
                possible_verbs[f"tam{keken}alt"] = [f"you {convert_to_past_tense(eng)}"]
                data.append(f"tas{keken}alt\tyou guys {convert_to_past_tense(eng)}\n")
                data.append(f"tas{keken}alt\tyou all {convert_to_past_tense(eng)}\n")
                possible_verbs[f"tas{keken}alt"] = [f"you guys {convert_to_past_tense(eng)}", f"you all {convert_to_past_tense(eng)}"]

                # future tense
                data.append(f"lok{keken}ku\tit will {eng}\n")
                possible_verbs[f"lok{keken}ku"] = [f"it will {eng}"]
                data.append(f"mep{keken}ku\tthis will {eng}\n")
                data.append(f"mep{keken}ku\tthat will {eng}\n")
                possible_verbs[f"mep{keken}ku"] = [f"this will {eng}", f"that will {eng}"]
                data.append(f"to{keken}ku\the will {eng}\n")
                possible_verbs[f"to{keken}ku"] = [f"he will {eng}"]
                data.append(f"ta{keken}ku\tshe will {eng}\n")
                possible_verbs[f"ta{keken}ku"] = [f"she will {eng}"]
                data.append(f"ma{keken}ku\tthey will {eng}\n")
                possible_verbs[f"ma{keken}ku"] = [f"they will {eng}"]
                data.append(f"me{keken}ku\tthey will {eng}\n")
                possible_verbs[f"me{keken}ku"] = [f"they will {eng}"]
                data.append(f"se{keken}ku\tthey will {eng}\n")
                possible_verbs[f"se{keken}ku"] = [f"they will {eng}"]
                data.append(f"ko{keken}ku\tI will {eng}\n")
                possible_verbs[f"ko{keken}ku"] = [f"I will {eng}"]
                data.append(f"tam{keken}ku\tyou will {eng}\n")
                possible_verbs[f"tam{keken}ku"] = [f"you will {eng}"]
                data.append(f"tas{keken}ku\tyou guys will {eng}\n")
                data.append(f"tas{keken}ku\tyou all will {eng}\n")
                possible_verbs[f"tas{keken}ku"] = [f"you guys will {eng}", f"you all will {eng}"]

        #console.print(possible_nouns)
        #console.print(possible_adjectives)
        
        progress.update(task1, total=1, completed=1)
        task2 = progress.add_task("Calculating number of sentence templates..", total=None)
        console.print("[d]Calculating total number of phrase templates..")

        # calculate number of templates
        num_templates = 0

        # Predefine the tag checks once
        TAGS = {
            "noun": ["{noun}", "{plural_noun}", "{definite_noun}", "{definite_plural_noun}"],
            "adjective": ["{adjective}"],
            "verb": ["{verb}"],
            "adverb": ["{adverb}"]
        }

        for keken_template, english_template in sentence_templates.items():
            # Determine which parts are needed
            needs = {
                "noun": any(tag in keken_template for tag in TAGS["noun"]),
                "adjective": any(tag in keken_template for tag in TAGS["adjective"]),
                "verb": any(tag in keken_template for tag in TAGS["verb"]),
                "adverb": any(tag in keken_template for tag in TAGS["adverb"]),
            }

            # Choose appropriate sources or fallback defaults
            noun_items = nouns.items() if needs["noun"] else [("", [""])]
            adj_items = possible_adjectives.items() if needs["adjective"] else [("", [""])]
            verb_items = possible_verbs.items() if needs["verb"] else [("", [""])]
            adv_items = adverbs.items() if needs["adverb"] else [("", [""])]

            # Loop through combinations
            for (keken_noun, eng_nouns), (keken_adj, eng_adjs), (keken_verb, eng_verbs), (keken_adv, eng_advs) in product(
                noun_items, adj_items, verb_items, adv_items
            ):
                for eng_noun, eng_adj, eng_verb, eng_adv in product(eng_nouns, eng_adjs, eng_verbs, eng_advs):
                    num_templates += 1
                    progress.update(task2, description="Calculating number of sentence templates.. [d](" + str(num_templates) + ")")
        # done calculating number of templates
        
        progress.update(task2, total=1, completed=1)
        task3 = progress.add_task("Fill in sentence templates..", total=num_templates)
        for keken_template, english_template in sentence_templates.items():
            needs_noun = any(tag in keken_template for tag in ["{noun}", "{plural_noun}", "{definite_noun}", "{definite_plural_noun}"])
            needs_adj = "{adjective}" in keken_template
            needs_verb = "{verb}" in keken_template
            needs_adv = "{adverb}" in keken_template

            for keken_noun, eng_nouns in (nouns.items() if needs_noun else [("", [""])]):
                for keken_adj, eng_adjs in (possible_adjectives.items() if needs_adj else [("", [""])]):
                    for keken_verb, eng_verbs in (possible_verbs.items() if needs_verb else [("", [""])]):
                        for keken_adv, eng_advs in (adverbs.items() if needs_adv else [("", [""])]):
                            for eng_noun, eng_adj, eng_verb, eng_adv in product(eng_nouns, eng_adjs, eng_verbs, eng_advs):
                                # --- Keken side ---
                                keken_sentence = keken_template
                                keken_sentence = keken_sentence.replace("{noun}", keken_noun)
                                keken_sentence = keken_sentence.replace("{plural_noun}", f"we{keken_noun}")
                                keken_sentence = keken_sentence.replace("{definite_noun}", f"val{keken_noun}")
                                keken_sentence = keken_sentence.replace("{definite_plural_noun}", f"valwe{keken_noun}")
                                keken_sentence = keken_sentence.replace("{adjective}", keken_adj)
                                keken_sentence = keken_sentence.replace("{verb}", keken_verb)
                                keken_sentence = keken_sentence.replace("{adverb}", keken_adv)

                                # --- English side ---
                                english_sentence = english_template
                                english_sentence = english_sentence.replace("{noun}", eng_noun)
                                english_sentence = english_sentence.replace("{plural_noun}", p.plural(eng_noun))
                                english_sentence = english_sentence.replace("{definite_noun}", f"the {eng_noun}")
                                english_sentence = english_sentence.replace("{definite_plural_noun}", f"the {p.plural(eng_noun)}")
                                english_sentence = english_sentence.replace("{adjective}", eng_adj)
                                english_sentence = english_sentence.replace("{verb}", eng_verb)
                                english_sentence = english_sentence.replace("{adverb}", eng_adv)

                                # Add only if all placeholders are replaced
                                if "{" not in keken_sentence and "{" not in english_sentence:
                                    data.append(f"{keken_sentence}\t{english_sentence}\n")
                                    progress.advance(task3)

        task4 = progress.add_task("Sorting..", total=None)
        data = sorted(data, key=len)
        progress.update(task4, total=1, completed=1)

        task5 = progress.add_task("Saving..", total=None)

        final_lines = ''.join(data)[:-1]
        with open("english-keken.txt", "w", encoding="utf-8") as f:
            f.write(final_lines)

        progress.update(task5, total=1, completed=1)
    console.print(f"[d]Generating data took {round(time()-start, 1)} seconds.")

def create_datasets(args):
    with Progress() as progress:
        input_name = args.input
        output_name = args.output

        task1 = progress.add_task("Loading dataset document..",total=None)
        task2 = progress.add_task("Creating pairs..",total=None, start=False)
        task3 = progress.add_task("Loading cleaned pairs..",total=None, start=False)
        task4 = progress.add_task("Seperating datasets..",total=None, start=False)
        task5 = progress.add_task("Saving datasets..", total=3, start=False)

        doc = load_doc(f"{input_name}.txt")
        progress.update(task1, total=1,completed=1)
    
        progress.start_task(task2)
        progress.print("[d]Converting to pairs..")
        pairs = to_pairs(doc)
        progress.print("[d]Cleaning pairs..")
        cleaned_pairs = clean_pairs(pairs)
        progress.print("[d]Saving cleaned pairs..")
        save_clean_data(cleaned_pairs, f"{output_name}.pkl")

        progress.update(task2, total=1,completed=1)
        progress.start_task(task3)
        
        raw_data = load_clean_sentences(f"{output_name}.pkl")
        progress.update(task3, total=1,completed=1)
        progress.start_task(task4)
        
        n_sentences = len(raw_data)

        dataset = raw_data[:n_sentences, :]
        progress.print("[d]Shuffling data..")
        # shuffle data
        shuffle(dataset)
        train, test = dataset[:int(n_sentences*args.train_percent)], dataset[int(n_sentences*args.train_percent):]

        table = Table("[b]Total", "[b bright_green]Training", "[b bright_blue]Testing", title="Data Layout")
        table.add_row(str(n_sentences), f"[b bright_green]{len(train)}  ({round(args.train_percent*100, 1)}%)", f"[b bright_blue]{len(test)} ({round((1-args.train_percent)*100, 1)})%")
        console.print()
        console.print(table)
        progress.update(task4, total=1,completed=1)

        # save
        progress.start_task(task5)
        save_clean_data(dataset, f"{output_name}-both.pkl")
        progress.advance(task5)
        save_clean_data(train, f"{output_name}-train.pkl")
        progress.advance(task5)
        save_clean_data(test, f"{output_name}-test.pkl")
        progress.advance(task5)

def train(args):
    dataset_name = args.dataset_name
    output_name = args.output

    dataset = load_clean_sentences(f"{dataset_name}-both.pkl")
    train = load_clean_sentences(f"{dataset_name}-train.pkl")
    test = load_clean_sentences(f"{dataset_name}-test.pkl")

    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])

    kek_tokenizer = create_tokenizer(dataset[:, 1])
    kek_vocab_size = len(kek_tokenizer.word_index) + 1
    kek_length = max_length(dataset[:, 1])

    console.print("[b blue]=== VOCAB SIZES AND LENGTHS ===")
    console.print(f"English vocab size: {eng_vocab_size}")
    console.print(f"English max length: {eng_length}\n")

    console.print(f"Keken vocab size: {kek_vocab_size}")
    console.print(f"Keken max length: {kek_length}\n")

    console.print("[b green]=== TRAINING ===")
    # prepare training data
    console.print("[d]Preparing training data...", highlight=False)
    trainX = encode_sequences(kek_tokenizer, kek_length, train[:, 1])
    trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    trainY = encode_output(trainY, eng_vocab_size)

    # prepare validation data
    console.print("[d]Preparing validation data...", highlight=False)
    testX = encode_sequences(kek_tokenizer, kek_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_output(testY, eng_vocab_size)

    # Define the model
    console.print("[d]Defining model...", highlight=False)
    model = define_model(kek_vocab_size, eng_vocab_size, kek_length, eng_length, args.n_units)
    # compile the model
    console.print("[d]Compiling...", highlight=False)
    optimizer = Adam(learning_rate=args.training_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # fit model
    console.print("[d]Training...", highlight=False)
    filename = output_name
    checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(trainX, trainY, epochs=args.epochs, batch_size=args.batch_size, validation_data=(testX, testY), callbacks=[checkpoint, lr_scheduler, early_stopping], verbose=2)

    # Summarize defined model
    console.print("[b green]Done!")
    model.summary()

    if args.create_model_diagram:
        plot_model(model, to_file="model.png", show_shapes=True)

def evaluate(args):
    console.print("[b blue]=== EVALUATION ===")
    console.print("[d]Preparing to evaluate model...", highlight=False)

    dataset_name = args.dataset_name
    model_path = args.model_path
    
    # load datasets
    dataset = load_clean_sentences(f'{dataset_name}-both.pkl')
    train = load_clean_sentences(f'{dataset_name}-train.pkl')
    test = load_clean_sentences(f'{dataset_name}-test.pkl')
    # prepare english tokenizer
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    # prepare keken tokenizer
    kek_tokenizer = create_tokenizer(dataset[:, 1])
    kek_length = max_length(dataset[:, 1])
    # prepare data
    trainX = encode_sequences(kek_tokenizer, kek_length, train[:, 1])
    testX = encode_sequences(kek_tokenizer, kek_length, test[:, 1])

    console.print("\n[d]Evaluating model...", highlight=False)
    model = load_model(model_path)
    print("Train:")
    evaluate_model(model, eng_tokenizer, trainX, train)
    print("Test:")
    evaluate_model(model, eng_tokenizer, testX, test)

def translate(args):
    with console.status("Loading model...") as status:
        model = load_model(args.model_path)

        status.update("Creating tokenizers...")
        # load datasets
        dataset = load_clean_sentences(args.tokenizer_data_path)
        # prepare english tokenizer
        eng_tokenizer = create_tokenizer(dataset[:, 0])
        # prepare keken tokenizer
        kek_tokenizer = create_tokenizer(dataset[:, 1])
        kek_length = max_length(dataset[:, 1])

        sentence: str = args.sentence
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans("", "", punctuation))

        status.update("Translating...")
        sequence = encode_sequences(kek_tokenizer, kek_length, [sentence])
        #sequence.reshape((1, sequence.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, sequence)

        console.print(f"Input sentence: {sentence}")
        console.print(f"Translation: {translation}")

arg_parser = ArgumentParser()
subparsers = arg_parser.add_subparsers()

create_datasets_arg = subparsers.add_parser("create_datasets", help="Create PKL datasets from the English-Keken TXT dataset")
create_datasets_arg.add_argument("-i", "--input", type=str, default="english-keken")
create_datasets_arg.add_argument("-o", "--output", type=str, default="english-keken")
create_datasets_arg.add_argument("--train_percent", type=float, default=0.8, help="The percentage of the pairs used as training data rather than testing, represented as a decimal.")
create_datasets_arg.set_defaults(func=create_datasets)

train_arg = subparsers.add_parser("train", help="Train the model")
train_arg.add_argument("--epochs", type=int, default=30, help="Number of epochs to train the model. Warning: Large epoch numbers may lead to overfitting!")
train_arg.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Warning: Large batch sizes may lead to overfitting!")
train_arg.add_argument("--dataset_name", type=str, help="The dataset name.", default="english-keken")
train_arg.add_argument("--output", type=str, help="The output model filename.", default="english-keken.h5")
train_arg.add_argument("--create_model_diagram", action="store_true", help="Create an image called \"model.png\" showing the structure of the AI model when training is complete.")
train_arg.add_argument("--n_units", type=int, help="The size of the model, if your dataset is small, use a value like 128 or 64, large datasets need large sizes like 256.", default=256)
train_arg.add_argument("--training_rate", type=float, help="The training rate of the model, too high and the model is unstable and overfits, too low and it doesnt learn anything.", default=0.001)
train_arg.set_defaults(func=train)

translate_arg = subparsers.add_parser("translate", help="Translate sentences using the trained model")
translate_arg.add_argument("sentence", type=str, help="The sentence to translate")
translate_arg.add_argument("--model_path", type=str, help="File path to the model to load.", default="english-keken.h5")
translate_arg.add_argument("--tokenizer_data_path", type=str, help="Path to data used for tokenizing.", default="english-keken-both.pkl")
translate_arg.set_defaults(func=translate)

eval_arg = subparsers.add_parser("eval", help="Evaluate the model on the test set")
eval_arg.add_argument("--dataset_name", type=str, help="The name of the dataset.", default="english-keken")
eval_arg.add_argument("--model_path", type=str, help="Path to the trained translation model.", default="english-keken.h5")
eval_arg.set_defaults(func=evaluate)

generate_data_arg = subparsers.add_parser("generate_data", help="Generate a dataset using the words in Keken.")
generate_data_arg.set_defaults(func=generate_data)

args = arg_parser.parse_args()
if hasattr(args, 'func'):
    args.func(args)
else:
    print("No command specified. Use --help for usage information.")