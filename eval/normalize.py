"""Faithful Python 3 port of the official VQA answer normalization.

Source: GT-Vision-Lab/VQA, PythonEvaluationTools/vqaEvaluation/vqaEval.py
(fetched 2026-11-10, sha256 f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8).

Quirks kept on purpose so scores match the official code:
  * periodStrip uses the (buggy) pattern "(?!<=\\d)(\\.)(?!\\d)" verbatim.
  * The official code calls periodStrip.sub("", text, re.UNICODE); in that call re.UNICODE (=32)
    is the positional *count* argument, so at most 32 periods are removed. We pass count=32.
  * processPunctuation decides strip-vs-space per punctuation mark using the ORIGINAL input text.
"""
import re

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
         ">", "<", "@", "`", ",", "?", "!"]
_OFFICIAL_PERIOD_COUNT = 32  # int(re.UNICODE), passed positionally as `count` in the official code


def clean_whitespace(text: str) -> str:
    """Pre-step applied to every answer in the official evaluate()."""
    return text.replace("\n", " ").replace("\t", " ").strip()


def process_punctuation(in_text: str) -> str:
    out_text = in_text
    has_comma_digit = COMMA_STRIP.search(in_text) is not None
    for p in PUNCT:
        if (p + " " in in_text or " " + p in in_text) or has_comma_digit:
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    return PERIOD_STRIP.sub("", out_text, count=_OFFICIAL_PERIOD_COUNT)


def process_digit_article(in_text: str) -> str:
    out = []
    for word in in_text.lower().split():
        word = MANUAL_MAP.get(word, word)
        if word not in ARTICLES:
            out.append(word)
    out = [CONTRACTIONS.get(w, w) for w in out]
    return " ".join(out)


def normalize_answer(text: str) -> str:
    """Full normalization (whitespace + punctuation + digits/articles/contractions).

    Used for building the answer vocabulary and for soft targets. NOTE: the official scorer applies
    this only when the 10 GT answers are not all identical; see eval/vqa_eval.py for the exact logic.
    """
    return process_digit_article(process_punctuation(clean_whitespace(text)))
