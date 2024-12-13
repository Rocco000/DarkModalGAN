from pathlib import Path
import pandas as pd
import re
import time
import matplotlib.pyplot as plt

# To apply the lemmatization
#nltk_path = Path("./.venv/nltk_data")
#nltk_path.mkdir(exist_ok=True)
#nltk.download('punkt', download_dir=p)
#nltk.download('punkt_tab', download_dir=p)
from nltk.stem import WordNetLemmatizer

import langid # To detect the text language
from deep_translator import GoogleTranslator # To translate no english text
from symspellpy import SymSpell, Verbosity # For spell checking

from transformers import BertTokenizer # For tokenization

drug_terms = [
    "cannabidiol",
    "cannabidiolic acid",
    "cannabidivarin",
    "cannabigerol",
    "cannabinol",
    "concentrate",
    "ak47",
    "shake",
    "tetrahydrocannabinol",
    "tetrahydrocannabinolic acid",
    "rick simpson oil",
    "nandrolone phenylpropionate",
    "trenbolone",
    "boldenone",
    "turinabol",
    "dihydrotestosterone",
    "ligandrol",
    "nutrobal",
    "ostarine",
    "human chorionic gonadotropin",
    "human growth hormone",
    "clostebol",
    "nandrolone",
    "androstenedione",
    "dimethyltryptamine",
    "lysergic acid diethylamide",
    "isolysergic acid diethylamide",
    "metaphedrone",
    "mephedrone",
    "nexus",
    "psilacetin",
    "mebufotenin",
    "psilocin",
    "methylenedioxymethamphetamine",
    "amphetamine",
    "methamphetamine",
    "oxycontin",
    "oxycodone",
    "acetylcysteine",
    # Drug name
    "k8",
    "rp15",
    "tramadol",
    "roxycodone",
    "nervigesic",
    "pregabalin",
    "carisoprodol",
    "alprazolam",
    "xanax",
    "anavar",
    "benzodiazepine",
    "cocaine",
    "clenbuterol",
    "benzocaine",
    "clomiphene",
    "crack",
    "marijuana",
    "hashish",
    "nbome",
    "hydroxycontin chloride",
    "ketamine",
    "heroin",
    "adderall",
    "sativa",
    "indica",
    "cookie",
    "mushroom",
    "dihydrocodeine",
    "psilocybin"
]

def initialize_bert_tokenizer() -> BertTokenizer:
    """
    Initialize the BertTokenizer for tokenization and add the domain-specific terms to the dictionary.
    :return: a BertTokenizer instance.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    new_tokens = [word for word in drug_terms if word not in tokenizer.vocab]

    tokenizer.add_tokens(new_tokens)

    return tokenizer

def tokenization(text:str, tokenizer:BertTokenizer) -> list:
    """
    Tokenize the input text using BertTokenizer.

    :param text: the input text.
    :param tokenizer: a BertTokenizer instance.
    :return: a list of tokens.
    """
    return tokenizer.tokenize(text)

def reconstruct_subwords(tokens:list) -> list:
    """
    Reconstruct full words from subword tokens (e.g., 'word' and '##s' -> 'words').

    :param tokens: a list of tokens.
    :return: a list of tokens without subword.
    """
    full_words = []
    current_word = ""
    for token in tokens:
        if token.startswith("##"):
            current_word += token[2:]  # Append subword (remove '##')
        else:
            if current_word:
                full_words.append(current_word)  # Save the current full word
            current_word = token  # Start a new word
    
    if current_word:
        full_words.append(current_word)
    return full_words

def initialize_symspell() -> SymSpell:
    """
    Initialize the SymSpell for spell checking and add the domain-specific terms to the dictionary.

    :return: a SymSpell instance.
    """
    # Initialize SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)

    dictionary_path = Path("./scripts/preprocessing_scripts/frequency_dictionary_en_82_765.txt")

    # Load a pre-built frequency dictionary
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    for word in drug_terms:
        sym_spell.create_dictionary_entry(word, 1)
    
    return sym_spell

def process_multi_word_terms(text:str) -> str:
    """
    Remove excess whitespace between domain-specific terms.

    :param text: the input text
    :return: the preprocessed text.
    """

    multi_word_terms_map = {
        r"cannabidiolic\s*acid": "cannabidiolic acid",
        r"tetrahydrocannabinolic\s*acid": "tetrahydrocannabinolic acid",
        r"rick\s*simpson\s*oil": "rick simpson oil",
        r"nandrolone\s*phenylpropionate": "nandrolone phenylpropionate",
        r"human\s*chorionic\s*gonadotropin": "human chorionic gonadotropin",
        r"human\s*growth\s*hormone": "human growth hormone",
        r"lysergic\s*acid\s*diethylamide": "lysergic acid diethylamide",
        r"isolysergic\s*acid\s*diethylamide": "isolysergic acid diethylamide",
        r"oxycodone\s*hydroxycontin\s*chloride": "oxycodone hydroxycontin chloride"
    }

    for multi_word, full_word in multi_word_terms_map.items():
        text = re.sub(multi_word, full_word, text)
    
    return text

def is_written_in_english(text:str) -> bool:
    """
    Detect the text language.

    :param text: the input text.
    :return: True whether the text language is English, False otherwise.
    """
    detected_lang = str(langid.classify(text)[0]).lower().strip()
    return True if detected_lang == "en" else False

def chunk_text(text:str, max_length=5000) -> list:
    """
    Split text into chunks of <= max_length characters without breaking words.

    :param text: the input text.
    :param max_length: a maximum length of a chunk (check the maximum length of the input text defined by the translator provider)
    :return: a list of chunks of the given text.
    """
    words = text.split()

    chunks = list()
    chunk = list()

    for word in words:
        length =  sum(len(w) + 1 for w in chunk) + len(word) + 1 # len of current chunk + len current word + 1 for whitespace

        if  length > max_length:
            chunks.append(" ".join(chunk))
            chunk = [word]
        else:
            chunk.append(word)

            if re.search(r"[\.!?]+$",word): # Whether the word finishes with a dot
                chunks.append(" ".join(chunk))
                chunk = []

    # Add the final chunk, if any    
    if len(chunk) > 0:
        chunks.append(" ".join(chunk))
    
    return chunks

def translate_text(text:str, retries=3) -> tuple[str,bool]:
    """
    Translate the input text in english.

    :param text: the input text.
    :param retries: the maximum number of attempts to translate the input text.
    :return: the translated text and a flag indicating whether the translation has been applied.
    """
    text = text.strip()

    chunks = chunk_text(text)
    translation = list()
    untranslated_flag = False
    
    for chunk in chunks:
        for attempt in range(retries):
            try:
                translated_chunk = GoogleTranslator(source="auto", target="en").translate(chunk)

                if translated_chunk is None or translated_chunk == "":
                    raise ValueError("Google Translator returns an empy string or a NoneType")

                translation.append(translated_chunk)
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed. Error message: {e}")
                print("Retry in 1 minute...")
                time.sleep(1)

                if attempt == 2:
                    untranslated_flag = True

    return " ".join(translation), untranslated_flag

def spell_checking(text:str, sym_spell:SymSpell, tokenizer:BertTokenizer) -> str:
    """
    Apply SymSpell to correct typing errors in the input text.

    :param text: the input text.
    :param tokenizer: a BertTokenizer instance.
    :return: the preprocessed text with correct words.
    """
    text = process_multi_word_terms(text) # Process multi-word terms

    tokens = tokenization(text, tokenizer)  # Tokenize text
    tokens = reconstruct_subwords(tokens)

    corrected_tokens = []
    for token in tokens:
        if token in drug_terms:
            corrected_tokens.append(token)
        else:
            suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=1)
            corrected_tokens.append(suggestions[0].term if suggestions else token)

    return " ".join(corrected_tokens)

def replace_concentration(text: str) -> str:
    """
    Standardize concentration patterns in product titles like '100 mg/ml' or '250 mg/1 ml' or '24 tb/50 mg'.

    :param text: the input text.
    :return: the preprocessed text with concentration patterns converts in a more readable format.
    """
    # Handle explicit and implicit "per" units
    pattern = re.compile(r"(\d+)(\.\d+)?\s*(μg|ug|mcg|mg|gr|g|kg|tab|tb|iu)\s*(?:/|in|per)\s*(\d+)?(\.\d+)?\s*(μg|mcg|mg|gr|g|kg|ml|l|tab|tb)")

    def replace_conc(match):
        # Extract matched groups
        amount1, decimal1, unit1, amount2, decimal2, unit2 = match.groups()

        txt = f"{amount1}"

        if decimal1:
            txt += f".{decimal1}"

        if amount2:
            txt += f" {unit1} per {amount2}"
            if decimal2:
                txt += f".{decimal2}"
            return f"{txt} {unit2}"
        return f"{txt} {unit1} per {unit2}"

    text = pattern.sub(replace_conc, text)

    # Approximation patterns (~)
    pattern = re.compile(r"~\s*(\d+)(\.\d+)?(?=\s*(μg|ug|mcg|mg|gr|g|kg|ml|l|oz|lb|lbs|tab|tb|iu))")

    def replace_conc2(match):
        # Extract matched groups
        amount1 = match.group(1)
        decimal1 = match.group(2)

        return f"approximately {amount1} " if not decimal1 else f"approximately {amount1}.{decimal1} "

    text = pattern.sub(replace_conc2, text)

    return text

def expand_acronyms(text:str) -> str:
    """
    Expand acronyms.

    :param text: the input text
    :return: the preprocessed text with expanded acronyms
    """
    acronym_mapping = {
        r"a\.?k\.?a\.?": "also known as" ,
        r"no.\s*(\d+)": "number \1",
        r"aaa": " "
    }

    for acronym, full_word in acronym_mapping.items():
        text = re.sub(acronym, full_word, text)
    
    return text

def expand_unit_measurement_acronyms(text: str) -> str:
    """
    Replace unit measurement acronyms in the input text using a defined mapping dictionary.
    
    :param text: the input text (title or description).
    :return: the processed text with acronyms replaced.
    """
    # Dictionary to map acronyms to full words
    unit_measurement_mapping = {
        r"(\d+(\.\d+)?)\s*(mcg|μg|ug)(?=\W|$)": r"\1 micrograms",
        r"(\d+(\.\d+)?)\s*mg(?=\W|$)": r"\1 milligrams",
        r"(\d+(\.\d+)?)\s*(grs?|g)(?=\W|$)": r"\1 grams", # catch 100gr and 100g
        r"(\d+(\.\d+)?)\s*kg(?=\W|$)": r"\1 kilograms",
        r"(\d+(\.\d+)?)\s*(lb|lbs)(?=\W|$)": r"\1 pounds",
        r"(\d+(\.\d+)?)\s*oz(?=\W|$)": r"\1 ounces",
        r"(\d+(\.\d+)?)\s*l(?=\W|$)": r"\1 liters",
        r"(\d+(\.\d+)?)\s*(ml|cc)(?=\W|$)": r"\1 milliliters",
        r"(\d+(\.\d+)?)\s*iu(?=\W|$)": r"\1 international unit",
        r"(\d+(\.\d+)?)\s*(?:inc|inch)(?=\W|$)": r"\1 inches",
        r"(\d+/\d+)\s*(?:inc|inch)(?=\W|$)": r"\1 inches",
    }
    for acronym, full_word in unit_measurement_mapping.items():
        text = re.sub(acronym, full_word, text)
    return text

def expand_drug_acronym(text: str) -> str:
    """
    Expand drug acronyms in the input text using a defined mapping dictionary.

    :param text: the input text.
    :return: the preprocessed text with expanded drug acronym.
    """
    acronym_mapping = {
        # Cannabis
        r"cbd": "cannabidiol",
        r"cbda": "cannabidiolic acid",
        r"cbdv": "cannabidivarin",
        r"cbg": "cannabigerol",
        r"cbn": "cannabinol",
        r"thc|thc-?o": "tetrahydrocannabinol",
        r"thca": "tetrahydrocannabinolic acid",
        r"rso": "rick simpson oil",
        r"ganja": "marijuana",
        r"weed": "marijuana",
        r"hash": "hashish",
        r"edibles?": "edible",

        # Steroids
        r"npp": "nandrolone phenylpropionate",
        r"tbol": "turinabol",
        r"lgd-4033": "ligandrol",
        r"mk-677": "nutrobal",
        r"mk-2866": "ostarine",
        r"hcg": "human chorionic gonadotropin",
        r"hgh": "human growth hormone",
        r"4-chlorotestosterone": "clostebol",
        r"19-nortestosterone": "nandrolone",
        r"andro": "androstenedione",

        # Psychedelics
        r"(n\s*(,|-)?\s*n)?\s*(,|-)?\s*dmts?": "dimethyltryptamine",
        r"(?<!iso-)lsd": "lysergic acid diethylamide",
        r"iso-lsd": "isolysergic acid diethylamide",
        r"(mu)?shrooms?": "mushroom",
        r"(?:\W|^)m\s*hrooms?": " mushroom",
        r"3-mmc": "metaphedrone",
        r"4-mmc": "mephedrone",
        r"2c-b": "nexus", #4-bromo-2.5-dimethoxyphenethylamine
        r"4-aco-dmt": "psilacetin",
        r"5-meo-dmt": "mebufotenin",
        r"4-ho-dmt": "psilocin",

        #ecstasy
        r"xtcy?": "ecstasy",
        r"mdma": "methylenedioxymethamphetamine",
        r"molly": "methylenedioxymethamphetamine",

        # Stimulants
        r"speed": "amphetamine",
        r"metha?": "methamphetamine",

        # Opioids
        r"oc": "oxycontin",
        r"diacetylmorphine": "heroin",
        r"diamorphine": "heroin",

        r"bh": " ",
        r"nac": "acetylcysteine", # n-acetylcysteine
        # Drug name
        r"ak\s*(-\s*)?47": "ak47",
        r"benzos?": "benzodiazepine",
        r"(25)?(\s*(i|x)\s*)?(\s*-\s*)?nbome?": "nbome",
        r"(clomifene|clomiphene)(\s*citrate)?": "clomiphene",
        r"coke": "cocaine",
        r"rp\s*15": "rp15",
        r"k\s*8": "k8",
        r"cookies": "cookie"
    }

    for acronym, full_name in acronym_mapping.items():
        text = re.sub(acronym+r"(?=\W|$)", full_name, text)
    
    return text

def replace_quantity_annotation(text: str) -> str:
    """
    Replace quantity notation like 'tabs', 'amps', 'caps', 'pcs', '10mg x 10 pills', '10mg x 10', '1x1mg', '1x', 'X100'.
    
    :param text: The input text.
    :return: the processed text with quantity notations standardized.
    """
    # Extend units' acronym
    unit_mapping = {
        r"(\d+(\.\d+)?)\s*(tabs?|tb)": r"\1 tablets",
        r"(\d+)\s*amps?": r"\1 ampoules",
        r"(\d+(\.\d+)?)\s*caps?": r"\1 capsules",
        r"(\d+)\s*pcs?": r"\1 pieces",
        r"(\d+)\s*drops": r"\1 drops"
    }

    for acronym, full_word in unit_mapping.items():
        text = re.sub(acronym+r"(?=\W|$)", full_word, text)

    # List of product words that indicate units
    product_units = r"(pills|tablets|capsules|vials|blister|bottles|ampoules)"

    # Replace patterns like "10mg x 10 pills", "10mg x 10"
    def replace_quantity1(match):
        # Extract matched groups
        amount1, decimal1, unit_measurament, amount2, unit = match.groups()
        txt = f"{amount1}"

        if decimal1 is not None:
            txt = f"{amount1}.{decimal1}"

        if unit is None:
            return f"{txt} {unit_measurament} per {amount2} pieces"
        else:
            return f"{txt} {unit_measurament} per {amount2} {unit}"

    text = re.sub(rf"(\d+)(\.\d+)?\s*(μg|ug|mg|gr|g|kg|lb|lbs|oz|l|ml)\s*x\s*(\d+)\s*{product_units}?", replace_quantity1, text)

    # Replace patterns like "1x1mg", "1x 1ml", "1x"
    def replace_quantity2(match):
        # Extract matched groups
        amount1, second_part_regex, amount2, decimal2, unit_mesurement = match.groups()

        # Case 1: No additional quantity expression after "x"
        if not second_part_regex:
            return f"{amount1} pieces"
        
        # Case 2: Handle quantity expression with amount and unit
        if not decimal2:
            return f"{amount1} pieces of {amount2} {unit_mesurement}"
        else:
            return f"{amount1} pieces of {amount2}.{decimal2} {unit_mesurement}"

    text = re.sub(r"(\d+)\s*x\s*((\d+)(\.\d+)?\s*(μg|ug|mg|gr|g|kg|lb|lbs|oz|l|ml))?(?=\W|$)", replace_quantity2, text)
    
    # Replace patterns like "X100", "X50"
    matches = re.finditer(r"\bx\s*(\d+)(?=\W+)", text)
    for match in matches:
        before = text[:match.start()]
        number = match.group(1)
        if not re.search(r"\d+\s*(?:μg|ug|mg|gr|g|kg|lb|lbs|oz|l|ml)$", before):
            text = re.sub(re.escape(match.group(0)), f"{number} pieces", text)

    #text = re.sub(r"(?<!\d+\s*(?:μg|ug|mg|gr|g|kg|lb|lbs|oz|l|ml))\s*x\s*(\d+)\W*", r"\1 pieces", text)

    # Remove quantity ranges like '20x--50x', '20x-50x', and '20--50'
    # text = re.sub(r"\b(\d+)\s*x?-{1,}\s*(\d+)x?\b", "", text)

    # Remove quantity range like "[10x-10k]"
    text = re.sub(r"\[\d+\s*x\s*-\s*\d+\s*k\]", "", text) 

    # Remove quantity range like "10pills - 20pills"
    text = re.sub(rf"\d+\s*{product_units}\s*-+\s*\d+\s*{product_units}", "", text) 

    # Remove quantity range like "10pcs - 20pcs", "5x - 5000x", "5 - 5000x", "5 - 5000pcs", "5 - 50", "5pcs - 10", "5x -10"
    text = re.sub(r"(\d+)\s*(pcs?|x)?\s*-+\s*(\d+)\s*(pcs?|x)?(?=\W|$)", "", text) 

    # Remove noisy patterns like "10x x10"
    text = re.sub(r"\b(\d+)x\s*x(\d+)\b", "", text)
    
    return text

def replace_percentage(text: str) -> str:
    """
    Replace '%' character like '70%', '50%' with 'percent'.

    :param text: the input text.
    :return: the preprocessed text without % character.
    """
    return re.sub(r"(\d+(\.\d+)?)\s*%", r"\1 percent ", text)

def remove_price(text: str) -> str:
    """
    Replace prices from the given string with an empty string.
    
    :param text: The input text.
    :return: the processed text without prices.
    """
    # Remove patterns like "10$", "10€", "10£"
    text = re.sub(r"\d+(\.\d+)?\s*[$€£]", " ", text)

    # Remove patterns like "$10", "€10", "£10"
    text = re.sub(r"[$€£]\s*\d+(\.\d+)?(?=\W|$)", " ", text)

    # Remove patterns like "10 euros", "10 dollars"
    text = re.sub(r"\d+(\.\d+)?\s*(euro|dollar)s?", " ", text)

    # Remove patterns like "10 usd"
    text = re.sub(r"\d+(\.\d+)?\s*usd", " ", text)

    return text

def remove_expiration_date(text: str) -> str:
    """
    Remove expiration dates from the input text such as 'exp 01/07', 'expires 01/07', '01/07'.

    :param text: the input text
    :return: the processed text without expiration dates.
    """
    return re.sub(r"(?:(?:exp(?:ires)?)?\d+[-\/\.]\d+(?:[-\/\.]\d+)?)", " ", text)

def remove_offer_word(text: str) -> str:
    """
    Remore words like 'offer', 'promo', etc.

    :param text: The input text.
    :return: the processed text without words as mentioned before.
    """
    promo_patterns = [
        r"special", r"(best|reduced|cheap|unbelievable|incredible)?\s*prices?", r"(limited\s*(time\s*)?|super)?offers?", r"promo",
        r"(exclusive\s*)?discount", r"intro", r"(super|hot|best)?\s*deal", r"extra", r"bargain",
        r"limited\s*time", r"exclusive", r"(flash|mega|super)?\s*sale", r"clearance", r"freebies?",
        r"(new)?in-?\s*stock", r"out\s*(of\s*)?stock"
    ]
    
    for promo_pattern in promo_patterns:
        pt = re.compile(promo_pattern)
        text = pt.sub("", text)
    
    return text

def remove_shipping_info(text: str, country_list: list):
    """
    Remove shipping information like 'free shipping', 'free ship' or origin and destination countries.

    :param text: the input text
    :param country_list: a list of countries to be removed from the input text.
    :return: the preprocessed text without shipping information.
    """
    # List of shipping patterns
    shipping_words = [
        r"shipping\s*free", r"free\s*shipping", r"free\s*ship", r"fast\s*shipping", r"shipping", r"shipped\s*free", r"ship(?:ped|ment)?", r"reship(?:ping)?", 
        r"(?:no)?\s*track(?:ing|ed)?", r"express", r"deliver(?:y|ed)(?:\s*time(?:s)?)?", r"order(?:ing|s)?",
        r"service", r"post(?:al)?", r"world\s*wide|worldwide", r"days", r"times", r"arrive", 
        r"intl|international", r"discreet", r"feedback", r"arrival", r"escrow"
    ]
    
    shipping_pattern = re.compile(r"\W*(" + "|".join(shipping_words) + r")\W*")
    text = shipping_pattern.sub(" ", text)

    # Remove countries
    country_patterns = re.compile(r"\W*(" + "|".join(map(re.escape, country_list)) + r")\W*")
    text = country_patterns.sub(" ", text, re.IGNORECASE)

    # Remove country acronym
    country_acronym_list = ["usa", "us", "u.s.", "u.s", "epigram", "eu", "aus", "can", "uk", "it", "ger", "fr", "nl"]

    acronym_patterns = re.compile(
        r"\W*(from)?(?:" + "|".join(map(re.escape, country_acronym_list)) +
        r")(?:\s*to\s*|\s*->\s*|[-\s]*)(?:" + "|".join(map(re.escape, country_acronym_list)) + r")?(?=\W|$)"
    )
    text = acronym_patterns.sub(" ", text)

    return text

def remove_vendor(text: str, vendor_list: list) -> str:
    """
    Remove vendor name from the input text.

    :param text: the input text.
    :param vendor_list: a list of vendors' name to be removed from the input text.
    :return: the preprocessed text without vendors' name.
    """
    for vendor in vendor_list:
        text = re.sub(rf"{vendor}", " ", text)

    text = re.sub(r"co\.?(?=\W+)", " ", text)

    text = re.sub(r"(vendor|seller)", " ", text)

    return text

def remove_macro_category(text: str) -> str:
    """
    Remove macro-categories from the input text.

    :param text: the input text.
    :return: the preprocessed text without macro-categories.
    """
    mc_list = [r"cannabis", r"dissociatives?", r"ecstasy", r"opioids?", r"psychedelics?", r"steroids?", r"stimulants?"]

    for pattern in mc_list:
        text = re.sub(pattern, " ", text)

    return text

def remove_repeated_words(text: str) -> str:
    """
    Remove consecutive repeated words in the input text.

    :param text: The input text.
    :return: The preprocessed text with repeated words removed.
    """
    # Match any repeated word and replace it with a single instance
    return re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text)

def remove_urls(text:str) -> str:
    """
    Delete any urls from the input text.

    :param text: the input text.
    :return: the preprocessed text without urls.
    """
    # Delete patterns like "http://", "https://", "www."
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)

    return text

def remove_unprintable_characters(text:str) -> str:
    """
    Remove unprintable characters fromt the input text.

    :param text: the input text.
    :return: the preprocessed text without unprintable characters.
    """
    # Replace non-printable or control characters
    return re.sub(r'[^\x20-\x7E]+', '', text)  # Only keep printable ASCII characters

def remove_special_characters(text: str) -> str:
    """
    Remove special characters like '!', '\', '(', etc. from the input text.

    :param text: The input text.
    :return: the processed text without special characters.
    """
    # Replace ? and ! with a dot
    text = re.sub(r"[?!]", ".", text)

    # Remove special characters (except dots)
    text = re.sub(r"[^a-zA-Z0-9.\s]", " ", text)
    
    # Replace multiple dots with a single dot
    text = re.sub(r"\.{2,}", ".", text)

    return text

def lemmatize_text(text:str, lemmatizer:WordNetLemmatizer, tokenizer:BertTokenizer) -> str:
    """
    Apply the Lemmatization to the input text which reduces words to their base or dictionary form.

    :param text: The input text.
    :param lemmatizer: a WordNetLemmatizer instance to apply lemmatization.
    :param tokenizer: a BertTokenizer instance.
    :return: lemmatized text.
    """
    text = process_multi_word_terms(text) # Process multi-word terms
    tokens = tokenization(text, tokenizer)  # Tokenize text
    tokens = reconstruct_subwords(tokens)

    lemmatized_tokens = []

    for token in tokens:
        if token in drug_terms:
            lemmatized_tokens.append(token)
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))

    return " ".join(lemmatized_tokens)

def preprocess_text(text: str, country_list: list, vendor_list: list, tokenizer:BertTokenizer, sym_spell:SymSpell, lemmatizer:WordNetLemmatizer) -> str:
    """
    Preprocess text.

    :param text: The input text.
    :param country_list: a list of countries.
    :param vendor_list: a list of vendors' name.
    :param tokenizer: a BertTokenizer instance for tokenization.
    :param sym_spell: a SymSpell instance for spell checking.
    :param lemmatizer: a WordNetLemmatizer instance for lemmatization.
    :return: the processed text.
    """
    # Lowercase
    text = text.lower()
    # To avoid error due to unprintable ASCII characters and not handled by Google Translator
    text = remove_price(text) 
    text = remove_urls(text)
    # Replace phase
    text = expand_acronyms(text)
    text = expand_drug_acronym(text)
    text = replace_concentration(text)
    text = replace_quantity_annotation(text)
    text = replace_percentage(text)
    text = expand_unit_measurement_acronyms(text)

    text = remove_unprintable_characters(text)

    if not is_written_in_english(text):
        text, untranslated_flag = translate_text(text)

        if untranslated_flag:
            print(f"Sorry but Google Translator hasn't been able to translate this text:\n{text}\n")

            text = str(input("Please provide the translated text:\n")).strip().lower()
            # Replace phase
            text = expand_acronyms(text)
            text = expand_drug_acronym(text)
            text = replace_concentration(text)
            text = replace_quantity_annotation(text)
            text = replace_percentage(text)
            text = expand_unit_measurement_acronyms(text)

    # Spell correction
    #text = spell_checking(text, sym_spell, tokenizer)

    # Remove phase
    text = remove_price(text)
    text = remove_expiration_date(text)
    text = remove_offer_word(text)
    text = remove_vendor(text, vendor_list)
    text = remove_shipping_info(text, country_list)
    text = remove_macro_category(text)

    text = remove_special_characters(text)

    # Replace one or more whitespace with one whitespace
    text = re.sub(r"\s+", " ", text)

    text = text.strip() # Remove whitespace at the start and at the end of the string

    # Lemmatization
    text = lemmatize_text(text, lemmatizer, tokenizer)

    return text
    
def merge_title_description(data:pd.DataFrame) -> pd.DataFrame:
    """
    Merge title and description fields in a new field.

    :param data: a DataFrame containing the crawled data.
    :return: a DataFrame with a new field which contains title and description.
    """
    data["full_text"] = data["title"] + ". " + data["description"]

    return data

def split_dataset(data:pd.DataFrame, max_token_length:int, save_path:Path) -> None:
    """
    Split the input dataset in two subdatasets based on max_token_length.

    :param data: a DataFrame containing the crawled data.
    :param max_token_length: maximum token lenght.
    """

    # Split the data based on max_token_length
    data_le = data[data['token_length'] <= max_token_length][["image_path", "title", "description", "full_text", "token_length"]]
    data_gt = data[data['token_length'] > max_token_length][["image_path", "title", "description", "full_text", "token_length"]]

    # Save the subsets to CSV files
    le_file = save_path.joinpath("text_analysis/text_le_e.csv")
    gt_file = save_path.joinpath("text_analysis/text_gt.csv")
    
    data_le.to_csv(le_file, index=False)
    data_gt.to_csv(gt_file, index=False)

def plot_full_text_length(data: pd.DataFrame, save_path: Path) -> None:
    """
    Plot the length of the full_text field.

    :param data: DataFrame containing the crawled data.
    :param save_path: a path where CSV files and plots will be stored.
    """
    text_analysis_path = save_path.joinpath("text_analysis")
    text_analysis_path.mkdir(exist_ok=True)

    # Plot the text length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data["token_length"], bins=50, color='blue', alpha=0.7)
    plt.axvline(512, color='red', linestyle='dashed', linewidth=1, label="512 Tokens") # Add a line to show the maximum token length
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Number of Texts")
    plt.legend()
    plt.savefig(text_analysis_path.joinpath("plot_token_length.png"))
    plt.show()

    print("Text length statistics:")
    stats = data["token_length"].describe()
    print(stats)

    # Convert stats to a DataFrame and save to CSV
    stats_df = stats.to_frame(name="Statistics").reset_index()  # Convert Series to DataFrame
    stats_df.columns = ["Metric", "Value"]  # Rename columns
    stats_df.to_csv(text_analysis_path.joinpath("token_length_statistics.csv"), index=False)

def main_preprocessing_text_data(data: pd.DataFrame, save_path:Path, country_list:list) -> pd.DataFrame:
    """
    The main function to preprocess the textual data.

    :param data: DataFrame containing the crawled data.
    :param save_path: a Path where analysis results will be stored.
    :param country_list: a list of countries to be removed from textual data.
    :return: a preprocessed DataFrame.
    """
    print("------------Preprocessing textual data------------")
    data_copy = data.copy()

    # Extract all vendors
    vendors = data_copy["vendor"].unique()

    tokenizer = initialize_bert_tokenizer()
    sym_spell = initialize_symspell()
    lemmatizer = WordNetLemmatizer()

    # Preprocess title
    print("Processing title...")
    data_copy["title"] = data_copy["title"].apply(lambda x: preprocess_text(x, country_list, vendors, tokenizer, sym_spell, lemmatizer) if isinstance(x, str) else x)

    # Preprocess description
    print("Processing description...")
    data_copy["description"] = data_copy["description"].apply(lambda x: preprocess_text(x, country_list, vendors, tokenizer, sym_spell, lemmatizer) if isinstance(x, str) else x)

    data_copy = merge_title_description(data_copy)

    # Calculate token length
    data_copy["token_length"] = data_copy["full_text"].apply(lambda x: len(tokenization(x, tokenizer)) if isinstance(x, str) else 0)

    plot_full_text_length(data_copy, save_path)

    split_dataset(data_copy, 510, save_path) # 512 - 2  to include special tokens such as CLS and SEP

    return data_copy