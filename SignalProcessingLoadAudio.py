import sys,os 
import torch, torchaudio
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import folder_paths

class SignalProcessingLoadAudio():
    supported_formats = ['wav','mp3','ogg','m4a','flac','mp4']
    @classmethod
    def INPUT_TYPES(s):
        supported_extensions = tuple(f".{fmt.lower()}" for fmt in SignalProcessingLoadAudio.supported_formats)
        
        input_dir = folder_paths.get_input_directory()
        all_items = os.listdir(input_dir)
        filtered_files = [
            x for x in all_items
            if x.lower().endswith(supported_extensions)
        ]
        files = [os.path.join(input_dir,x) for x in filtered_files]

        return {
            "required":  {"audio_file": (sorted(files), {"image_upload": True})},
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0})}
        }

    RETURN_TYPES = ("AUDIO","INT")
    RETURN_NAMES = ("audio","sample_rate")
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(self, audio_file, seek_seconds):

        audio_file_path = folder_paths.get_annotated_filepath(audio_file)

        waveform, sample_rate = torchaudio.load(audio_file_path)
        waveform = waveform.unsqueeze(0).contiguous()

        print('SignalProcessingLoadAudio.waveform',waveform.shape)
        return {'waveform': waveform, 'sample_rate': sample_rate} , sample_rate

    @classmethod
    def IS_CHANGED(s, audio_file, seek_seconds):
        audio_file_path = folder_paths.get_annotated_filepath(audio_file)
        m = hashlib.sha256()
        with open(audio_file_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, seek_seconds):
        if not folder_paths.exists_annotated_filepath(audio_file):
            return "Invalid image file: {}".format(audio_file)

        return True