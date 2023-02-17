# Author: fforfabio

##########################################################################
# FFT Script

# EN: dependencies
# IT: Dipendenze
import time
import numpy
from scipy.fft import fft


# EN: Wrapper function used to calculate the moving average on
# an odd or even window

# IT: Funzione d'interfaccia per calcolare la media mobile su 
# una finestra di dimensioni pari o dispari.
def mov_mean(arr, win):
    n = len(arr)
    if win % 2 == 0:
        return mov_mean_even(arr, win, n)
    else:
        return mov_mean_odd(arr, win, n)


# EN: The matlab moving average with a window
# of odd size is calculated as follows:
# the window is centered on the i-th value.
# If the window doesn't cover its entirety, the same is calculated
# the moving average, but reducing the samples on which it is calculated.

# IT: La moving average di matlab con una finestra
# di dimensione dispari è calcolata nel seguente modo:
# la finestra è centrata sull'i-esimo valore.
# Se la finestra non copre tutta la sua interezza si calcola lo stesso
# la moving average, riducendo però i campioni sulla quale si calcola.
def mov_mean_odd(arr, win, l):
    movmean_ = []
    for i in range(l):
        # EN: Incomplete window on the left
        # IT: Finestra incompleta sulla sinistra
        if i < win//2:
            # EN: sum(array[0:i + (win//2) + 1] sums the values ​​stored in array
            # from index 0 to index i + (win//2) (that is the middle of the window) +1
            # Other similar data recovery instructions should be interpreted the same way.
            # The i-th value of the moving average (sum / window size) is added to the array
            
            # IT: sum(array[0:i + (win//2) + 1] effettua la somma dei valori memorizzati in array
            # dall'indice 0 all'indice i + (win//2) (ovvero la metà della finestra) +1
            # Così sono da interpretare anche le altre istruzioni simili di recupero dati.
            # Si aggiunge all'array il valore i-esimo della moving average (somma / dimensione finestra)
            movmean_.append(sum(arr[0:i + (win//2) + 1]) / (i + (win//2) + 1))
        # EN: Incomplete window on the right
        # IT: Finestra incompleta sulla destra
        elif l <= (i + win//2):
            movmean_.append(sum(arr[i - (win//2):l]) / (l - i + win//2))
        # EN: Complete window
        # IT: Finestra completa
        else:
            movmean_.append(sum(arr[i - (win//2):i + (win//2) + 1]) / win)
    return movmean_



# EN: The matlab moving average with a window
# of even size (in this case 20), is calculated as follows:
# the window is centered on the i-th value and on the previous one.
# If the window doesn't cover its entirety, the same is calculated
# the moving average, but reducing the samples on which it is calculated.

# IT: La moving average di matlab con una finestra
# di dimensione pari (in questo caso 20), è calcolata nel seguente modo:
# la finestra è centrata sull'i-esimo valore e su quello precedente.
# Se la finestra non copre tutta la sua interezza si calcola lo stesso
# la moving average, riducendo però i campioni sulla quale si calcola.
def mov_mean_even(arr, win, l):
    movmean_ = []
    for i in range(l):
        # EN: Incomplete window on the left
        # IT:Finestra incompleta sulla sinistra
        if i < win//2:
            movmean_.append(sum(arr[0:i + (win//2)]) / (i + (win//2)))
        # EN: Incomplete window on the right
        # IT: Finestra incompleta sulla destra
        elif l < (i + win//2):
            movmean_.append(sum(arr[i - (win//2):l]) / (l - i + win//2))
        # EN: Complete window
        # IT: Finestra completa
        else:
            movmean_.append(sum(arr[i - (win//2):i + (win//2)]) / win)
    return movmean_
    
    
# EN: Function that finds peaks in every 1 Hz interval.
# Then the peak in 0-1, in 1-2, in 2-3 and in 3-4 Hz.

# IT: Funzione che trova i picchi in ogni intervallo di 1 Hz.
# Quindi il picco in 0-1, in 1-2, in 2-3 ed in 3-4 Hz.
def found_max_peak_for_freq_range(arr, freq):
    max_p, max_f, arr_peak, arr_freq, index = -1, -1, [], [], 0

    [max_p, max_f, index] = cycle_for_max(0, arr, freq, 1)
    arr_peak.append(max_p)
    arr_freq.append(max_f)

    [max_p, max_f, index] = cycle_for_max(index, arr, freq, 2)
    arr_peak.append(max_p)
    arr_freq.append(max_f)

    [max_p, max_f, index] = cycle_for_max(index, arr, freq, 3)
    arr_peak.append(max_p)
    arr_freq.append(max_f)

    [max_p, max_f, index] = cycle_for_max(index, arr, freq, 4)
    arr_peak.append(max_p)
    arr_freq.append(max_f)

    return arr_peak, arr_freq
    

# EN: Function that cycles through a range of frequencies and searches
# for the maximum peak.

# IT: Funzione che cicla su un intervallo di frequenze e cerca
# il picco massimo.
def cycle_for_max(start, arr, freq, threshold):
    max_p, max_f = -1, -1
    for i in range(start, len(freq)):
        if freq[i] >= threshold:
            break
        else:
            if max_p < arr[i]:
                max_p = arr[i]
                max_f = freq[i]
    return max_p, max_f, i
    
    
# EN: Calculation of the FFT and of the frequencies corresponding to the samples.
# The Matlab algorithm is used.

# IT: Calcolo della FFT e delle frequenze corrispondenti ai campioni.
# Si utilizza l'algoritmo di Matlab.
def FFT(raw_data, time_arr):
    L, f = len(time_arr), []

    # EN: Calculate diff array, where d[i] = time[i+1] - time[i].
    # IT: Calcolo del vettore diff, dove d[i] = time[i+1] - time[i].
    d = numpy.diff(time_arr)
    Fs = 1 / (sum(d) / len(d))

    # EN: Calculate of the FFT on teh raw_data array
    # IT: Calcolo della FFT sul vettore raw_data
    raw_data_fft = fft(raw_data)

    # EN: abs on the normalized FFT
    # IT: Calcolo del valore assoluto sulla FFT normalizzata.
    abs_data = abs(raw_data_fft / L)

    # EN: Take only positive frequencies
    # IT: Si prendono le frequenze positive.
    P1 = abs_data[0:L//2+1]
    P1[0:len(P1) - 1] = 2*P1[0:len(P1) - 1]
    # for r in range(1, len(P1) - 1):
    #     P1[r] = 2*P1[r]

    # EN: Frequencies calculation
    # IT: Si calcolano le frequenze
    a = numpy.arange(0, L//2 + 1)
    # for r in range(L//2 + 1):
    #     f.append(Fs * r / L)
    f = Fs * a / L

    # EN: FFT plot
    # IT: Grafico della FFT
    # fig, ax = plt.subplots()
    # ax.plot(f, P1)
    # plt.grid()
    # plt.show()

    # [peaks, f_p] = found_peaks(P1, f)
    [peaks1, f_p1] = found_max_peak_for_freq_range(P1, f)
    # return P1, f, peaks, f_p, peaks1, f_p1
    return P1, f, peaks1, f_p1
    
    
def Check_peaks(p, fr, L, H):
    # EN: Check peaks acceptance criteria
    # IT: Verifica condizioni accettabilità
    for i in range(len(p)):
        if p[i] > H:
            print("BIG Problem: Rejection limit!!!\nPeak: " + str(round(p[i], 4)) + " Nm at "
                  + str(round(fr[i], 4)) + " Hz")
        elif L < p[i] < H:
            print("Problem: Monitoring limit!!!\nPeak: " + str(round(p[i], 4)) + " Nm at " +
                  str(round(fr[i], 4)) + " Hz")
        else:
            continue
            
            
def create_time(N, window):
    # EN: Creation of time array and then remove first and last 10 sample.
    
    # IT: Creazione del vettore dei tempi. E poi rimozione dei primi 10
    # e successivi 10 campioni.
    t = numpy.arange(0, 10, 0.005)
    t = t[(window // 2) - 1:N - ((window // 2) + 1)]
    return t