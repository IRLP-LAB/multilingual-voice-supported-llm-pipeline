import os
import re
import glob
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from vits.models import SynthesizerTrn
# from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from vits import utils
from scipy.io.wavfile import write
from data_utils.data_helpers import TextMapper, download
from scipy.io.wavfile import write



class MMSTTS():

    def __init__(self, language):
        self.language = language
        self.checkpoint_dir = download(language)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        vocab_file = f"{self.checkpoint_dir}/vocab.txt"
        config_file = f"{self.checkpoint_dir}/config.json"
        self.hps = utils.get_hparams_from_file(config_file)
        self.text_mapper = TextMapper(vocab_file)
        
        self.net_g = SynthesizerTrn(
            len(self.text_mapper.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        self.net_g.to(self.device)
        _ = self.net_g.eval()

        g_pth = f"{self.checkpoint_dir}/G_100000.pth"
        _ = utils.load_checkpoint(g_pth, self.net_g, None)

        
        
    
    # def initialize_model(self):
    #     if torch.cuda.is_available():
    #         self.device = torch.device("cuda")
    #     else:
    #         self.device = torch.device("cpu")

    #     vocab_file = f"{self.checkpoint_dir}/vocab.txt"
    #     config_file = f"{self.checkpoint_dir}/config.json"
    #     self.hps = utils.get_hparams_from_file(config_file)
    #     self.text_mapper = TextMapper(vocab_file)
        
    #     self.net_g = SynthesizerTrn(
    #         len(self.text_mapper.symbols),
    #         self.hps.data.filter_length // 2 + 1,
    #         self.hps.train.segment_size // self.hps.data.hop_length,
    #         **self.hps.model)
    #     self.net_g.to(self.device)
    #     _ = self.net_g.eval()

    #     g_pth = f"{self.checkpoint_dir}/G_100000.pth"
    #     _ = utils.load_checkpoint(g_pth, self.net_g, None)

    def synthesize(self, txt):
        """
        Generate audio for the specified text using the TTS model.
        
        Parameters:
        - txt (str): The input text for which the audio needs to be synthesized.
        
        Returns:
        - audio (numpy.ndarray): The synthesized audio in numpy array format.
        """
        # Preprocess the text
        txt = self.preprocess_text(txt)
        stn_tst = self.text_mapper.get_text(txt, self.hps)

        # Synthesize audio
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            audio = self.net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0, 0].cpu().float().numpy()

        return audio
    
    # def preprocess_char(text, lang=None):            
    #     """
    #     Special treatment of characters in certain languages.
    #     """
    #     if lang == 'ron':
    #         text = text.replace("ț", "ţ")
    #     return text


    def preprocess_text(self, txt, lang=None):
        """
        Preprocess the input text for the model.
        
        Parameters:
        - txt (str): Input text.
        - lang (str, optional): Language of the input text.
        
        Returns:
        - str: Preprocessed text.
        """
        # txt = preprocess_char(txt, lang=lang)
        # If there's a need for uromanize in your use case, it can be called here
        txt = txt.lower()
        # Filter out-of-vocabulary characters using TextMapper's filter_oov method
        txt = self.text_mapper.filter_oov(txt)
        return txt

    def save_to_file(self, audio_array, filename='output.wav'):
            """
            Save the synthesized audio to a WAV file.
            
            Parameters:
            - audio_array (numpy.ndarray): The synthesized audio in numpy array format.
            - filename (str, optional): The name of the output WAV file.
            """
            write(filename, self.hps.data.sampling_rate, audio_array)