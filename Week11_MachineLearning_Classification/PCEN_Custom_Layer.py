'''
PCEN_Layer.py developed by: Egor Nenstel
'''

import torch
from torch import nn, sqrt
import torch.nn.functional as F
import math





class PCEN_Layer(nn.Module):
    # https://librosa.org/doc/main/generated/librosa.pcen.html
    def __init__(
            self, 
            mel_spectrogram,
            time_constant = 0.400,                                                                                                      # Input spectrogram  -->  E(t,f) / X(t,f)
            hop_length=160,                                                                                                             # Hop length of the spectrogram --> t
            s=0.025,                                                                                                                    # Smoothing coefficient for the filtering operation --> s (b)
            sr=16000,                                                                                                                   # Sample rate of the audio signal --> sr
            alpha=0.98,                                                                                                                 # Compression exponent --> alpha 
            delta=2.0,                                                                                                                  # Bias added to the numerator --> delta        
            r=0.5,                                                                                                                      # Power to which the smoothed spectrogram is raised --> r
            eps=1e-6                                                                                                                    # Small constant to avoid division by zero --> eps      
    ):
        super(PCEN_Layer, self).__init__()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
        device = torch.device("cpu")                            
        T = time_constant * sr / hop_length
        b = (torch.sqrt(torch.tensor(1 + 4 * T**2)) - 1) / (2 * T**2)
        #
        if len(mel_spectrogram.shape) == 4:
            # Wähle die Dimension, entlang der gearbeitet werden sollte bei 4 Dimensionen
            self.batch_size, self.channel, self.frequency, self.time = 0,1,2,3
        elif len(mel_spectrogram.shape) == 3:
            # Wähle die Dimension, entlang der gearbeitet werden sollte bei nur 3 Dimensionen
            self.channel, self.frequency, self.time = 0,1,2
        else:
            print("Wrong dimension, must be 4 or at least 3 dimensional!")
        #
        # Learnable parameters
        # Dimension = [Batch_Size, Channel, Frequency, Time]                                                                             # [100, 1, 40, 101]
        self.raw_b = nn.Parameter(torch.full([mel_spectrogram.shape[self.frequency]], float(b),device=device))                           # b        - Glättungskoeffizient für die Filterung
        self.raw_alpha = nn.Parameter(torch.full([mel_spectrogram.shape[self.frequency]], float(alpha),device=device))                   # alpha    - Kompressionsexponent für die PCEN-Formel
        self.raw_delta = nn.Parameter(torch.full([mel_spectrogram.shape[self.frequency]], float(delta),device=device))                   # delta    - Bias, der zum Zähler der PCEN-Formel hinzugefügt wird
        self.raw_r = nn.Parameter(torch.full([mel_spectrogram.shape[self.frequency]], float(r),device=device))                           # r        - Exponent, auf den das geglättete Spektrogramm in der PCEN-Formel angehoben wird
        self.eps = eps                                                                                                                   # eps      - Kleine Konstante, um Division durch Null zu vermeiden

    def forward(self, mel_Spectrogram):
        device = mel_Spectrogram.device
        previous_MelSpectrogram = torch.zeros_like(mel_Spectrogram[:,:,:,0])
        PCEN_equ = torch.zeros_like(mel_Spectrogram, device=device)
        PCEN = []
        #
        b = torch.sigmoid(self.raw_b).view(1, -1)
        alpha = torch.sigmoid(self.raw_alpha).view(1, -1)
        r = torch.sigmoid(self.raw_r).view(1, -1)
        delta = F.softplus(self.raw_delta).clamp(min=1.0, max=10.0).view(1, -1)
        #
        # Compute the filtered spectrogram M
        # Implement the filtering operation here
        for t in range(mel_Spectrogram.shape[-1]):
            t1 = torch.full([mel_Spectrogram.shape[self.frequency]], float(t/(t+1)))
            t2 = torch.full([mel_Spectrogram.shape[self.frequency]], float(1/(t+1)))
            #
            b1 = (torch.min(1-b,t1.to(device))).view(1, -1)                                                                              
            b2 = (torch.max(b,t2.to(device))).view(1, -1)                                                                                      
            #
            smoothed_MelSpectrogram = b1 * previous_MelSpectrogram + b2 * mel_Spectrogram[:,:,:,t]
            previous_MelSpectrogram = smoothed_MelSpectrogram
            #
            PCEN_equ = (mel_Spectrogram[:,:,:,t] / (self.eps + smoothed_MelSpectrogram) ** alpha + delta) ** r - delta ** r
            PCEN.append(PCEN_equ)
        PCEN = torch.stack(PCEN, dim=-1)

        return PCEN