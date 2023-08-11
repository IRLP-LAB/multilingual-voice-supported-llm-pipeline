import os
import subprocess
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import re
from vits import commons
import torch 

def download(lang, tgt_dir="./"):
    
    """
    Download the model checkpoints for the specified language.
    
    Parameters:
    - lang (str): The language code for which the model checkpoints need to be downloaded.
    - tgt_dir (str): The target directory where the downloaded files will be saved.
    
    Returns:
    - lang_dir (str): The directory where the model checkpoints for the specified language are saved.
    """
    
    lang_fn, lang_dir = os.path.join(tgt_dir, lang+'.tar.gz'), os.path.join(tgt_dir, lang)
    
    # Check if the directory already exists
    if os.path.exists(lang_dir):
        print(f"The directory for language {lang} already exists.")
        return lang_dir

    cmd = ";".join([
        f"wget https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz -O {lang_fn}",
        f"tar zxvf {lang_fn}"
    ])
    
    print(f"Download model for language: {lang}")
    subprocess.check_output(cmd, shell=True)
    print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
    return lang_dir


class TextMapper:
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}
    
    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def sequence_to_text(self, sequence):
        return "".join([self._id_to_symbol[s] for s in sequence])

    def filter_oov(self, txt):
        filtered = []
        for w in txt.split():
            if any([c not in self._symbol_to_id for c in w]):
                continue
            filtered.append(w)
        return " ".join(filtered)

    
    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm