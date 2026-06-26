# the RTISI algorithm estimates the missing phase of a spectrogram from the magnitude of the spectrogram and the previous time-domain signal.
# it is not relevant for the exam

import numpy as np

class CRTISI(object):
    
    def __init__(self, hs, AnalysisWindow, SynthesisWindow = None):
        self.__hs = hs
        self.__AnalysisWindow = AnalysisWindow
        if SynthesisWindow is None:
            self.__SynthesisWindow = AnalysisWindow
        else:
            self.__SynthesisWindow = SynthesisWindow
        self.__Buffer = None
        self.__WindowSum = None
        self.__MaxIter = 2
            
    def __UpdateBuffer(self, ColumnOfSpectrogram):
        if self.__Buffer is None:
            Lookahead = int(self.__AnalysisWindow.shape[0] / self.__hs - 1)
            self.__Buffer = np.zeros((ColumnOfSpectrogram.shape[0], 2*Lookahead+1), dtype=complex)
            self.__FFTLen = 2 * (self.__Buffer.shape[0] - 1)
        else:
            NumberOfColumns = self.__Buffer.shape[1]
            self.__Buffer[:, 0:NumberOfColumns-1] = np.copy(self.__Buffer[:, 1:NumberOfColumns])
        self.__Buffer[:, -1] = np.copy(ColumnOfSpectrogram)            
    
    def GetLatencyInSamples(self):
        return (self.__Buffer.shape[1] // 2) * self.__hs
    
    def __GetWindowSize(self):
        return self.__AnalysisWindow.shape[0]
    
    def __GetWindowSum(self):
        if self.__WindowSum is None:
            ws = self.__GetWindowSize()
            self.__WindowSum = np.zeros((self.__hs * (self.__Buffer.shape[1] - 1) + ws))
            for column in range(self.__Buffer.shape[1]):
                idx1 = column * self.__hs
                idx2 = idx1 + ws
                self.__WindowSum[idx1:idx2] += self.__AnalysisWindow * self.__SynthesisWindow
        return self.__WindowSum
    
    def __OverlapAdd(self):
        # transform back into time-domain
        x = np.fft.irfft(self.__Buffer, axis = 0)
        # overlap add
        ws = self.__GetWindowSize()
        y = np.zeros((self.__hs * (x.shape[1] - 1) + ws))
        for column in range(x.shape[1]):
            idx1 = column*self.__hs
            idx2 = idx1 + ws
            LocalBlock = x[0:ws, column] * self.__SynthesisWindow
            y[idx1:idx2] += LocalBlock
            if column == x.shape[1] // 2:
                result = LocalBlock
        y /= self.__GetWindowSum()
        return y, result
    
    def __UpdatePhases(self, y):
        for column in range(self.__Buffer.shape[1] // 2, self.__Buffer.shape[1]):
            idx1 = column*self.__hs
            idx2 = idx1 + self.__GetWindowSize()
            X = np.fft.rfft(y[idx1:idx2] * self.__AnalysisWindow, n = self.__FFTLen)
            #if column == self.__Buffer.shape[1] // 2:
            #    SNR = 10*np.log10(np.sum(np.abs(self.__Buffer)**2)/np.sum((np.abs(X) - np.abs(self.__Buffer[:, column]))**2))
            self.__Buffer[:, column] = np.abs(self.__Buffer[:, column]) * X / np.abs(X)
        #return SNR
    
    def ProcessNewColumnOfSpectrogram(self, ColumnOfSpectrogram):
        self.__UpdateBuffer(ColumnOfSpectrogram)
        y, result = self.__OverlapAdd()
        for iter in range(self.__MaxIter):
            self.__UpdatePhases(y)
            y, result = self.__OverlapAdd()
            # After evaluating the final result in the final iteration, no __UpdatePhases can be called, because from now on,
            # the result is fixed and given back to the user. Therefore, the corresponding phase of the corresponding column
            # in the self.__Buffer is no longer changeable.
            # Therefore the iteration order is:
            # first update phases second evaluate overlap add
        return result