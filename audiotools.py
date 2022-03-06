#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Toolbox for generating, modifying, and analyzing audio data.

@author: jamesbigelow at gmail dot com


"""

#### Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


def audio_attenuate( y, atten ):

    '''
    
    Attenuate (audio waveform) 
    
    INPUT -------
    y : audio signal (sound pressure waveform)
    atten : attenuation (dB)

    RETURN -------
    y : attenuated audio signal 
    
    '''
    
    y = y * 10 **(-atten/20)

    return y    


def audio_ramp( y, fs, ramp_time ):
    
    '''
    
    Apply cosine-squared ramps to beginning and end of audio waveform 
    
    INPUT -------
    y : audio signal (sound pressure waveform)
    fs : audio sample rate (Hz), e.g., 48e3
    ramp_time : total ramp duration (s), 0.005 - 0.02 s usually appropriate
    
    RETURN -------
    y : ramped audio signal
    
    '''

    ramp_samples = int( np.round( ramp_time * fs ) )
    ramp_envelope = 0.5 * ( 1 - np.cos(  (2 * np.pi * np.arange(0,ramp_samples) ) / (2*ramp_samples)   ) )
    
    if len( y.shape ) == 1:
        y[ 0:ramp_samples ] = y[ 0:ramp_samples ] * ramp_envelope
        y[ -ramp_samples: ] = y[ -ramp_samples: ] * np.flipud( ramp_envelope )
    else:    
        for ii in range( y.shape[1] ):
            y[ 0:ramp_samples, ii ] = y[ 0:ramp_samples, ii ] * ramp_envelope
            y[ -ramp_samples:, ii ] = y[ -ramp_samples:, ii ] * np.flipud( ramp_envelope )
    
    return y


def audio_write_wav( y, fs, fid ): 
    
    '''
    
    Write (audio waveform) to wav file
    
    INPUT -------
    y : (audio waveform) vector
    fs : audio sample rate (Hz), e.g., 48e3
    fid : filename of wavfile, e.g., 'demo_audio.wav'
    
    '''
    
    # Normalize if necessary to avoid clipping
    if abs(y).max() > 1:
        y = y / abs(y).max()
        
    scipy.io.wavfile.write( fid, int(fs), y.astype(np.float32) ) # scipy does not rescale audio data, assumes f32 format
    
    

def filt_convfft( x, y, delay, n=False, opt_truncate=True ):
    
    '''
    
    Discrete Convolution performed by using FFT 
    
    INPUT -------
    x : signal 1
    y : signal 2
    delay : filter delay correction
    n = fft length (default if not provided)
    opt_truncate : truncate to x.size + y.size -1 (true/false)
    
    RETURN -------
    f : convolution 
    
    '''
    
    if not n:
        n = int( 2 ** ( np.ceil( max( ( np.log10(x.size)/np.log10(2), np.log10(y.size)/np.log10(2) ) ) ) ) )
        opt_truncate = False

    sx = np.zeros( n*2, dtype=float )
    sy = np.zeros( n*2, dtype=float )
    sx[0:x.size] = x
    sy[0:y.size] = y
    
    ftmp = np.real( np.fft.ifft( np.fft.fft(sy) * np.fft.fft(sx) ) )
    f = ftmp[0+delay:n+delay]
    
    if opt_truncate:
        f = f[ 0: x.size + y.size - 1 ]

    return f    
    

def filt_bandpass( fs, f1, f2, tw, atten, opt_plot=False ):

    '''
    
    Band pass filter
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    f1 : low frequency cutoff (Hz)
    f2 : high frequency cutoff (Hz)
    tw : transition width (Hz)
    atten : pass attenuation 
    opt_plot : optional plot of filtered power spectrum (true/false)
    
    RETURN -------
    h : impulse response 
    
    Example: 
        fs = 12500
        f1 = 85.28622364622576
        f2 = 121.84678710543922
        tw = 3.6560563459213458
        atten = 60

    '''
    
    
    if f1 == 0: 
        h = filt_lowpass( fs, f2, tw, atten )
    else:
        fc = ( f2 - f1 ) / 2
        h = filt_lowpass( fs, fc, tw, atten )
        fc0 = ( f2 - f1 ) / 2 + f1
        n = ( h.size - 1 ) / 2
        h = 2 * h * np.cos( 2 * np.pi * fc0 * ( np.arange( -n, n+1) ) / fs )
    
    if opt_plot:
        
        m = int( 2 ** np.ceil( np.log2( h.size ) ) * 16 )
        hh = abs( np.fft.fft(h,m) )
        ps = 20 * np.log10(hh)
        
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches( 5, 7.5 )
        
        ax[0].plot( np.arange( 1, m+1 )/m*fs, ps, color='k' )
        ax[0].set_xlim( ( 0, fs/2 ) )
        ax[0].set_ylim( ( min( ps[np.isfinite(ps)] ), 0 ) )
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Power spectrum (dB)')
        
        ax[1].plot( np.arange( 1, m+1 )/m*fs, hh, color='k' )
        ax[1].set_xlim( ( f1, f2 ) )
        ax[1].set_ylim( ( 1-10**(-atten/20), 1+10**(-atten/20) ) )
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Linear passband amplitude (dB)') 
    
    return h


def filt_design( fcr, twr, atten ):
    
    '''
    
    Filter design
    
    INPUT -------
    fcr : frequency cutoff (radians, 0 : pi)
    twr : transition width (radians, 0 : pi)
    atten : attenuation (dB)
    
    RETURN -------
    
    n : filter length 
    alpha : filter shape parameter
    p : filter smoothing parameter 
    
    Example: 
        fcr = 0.37699111843077515
        twr = 0.09424777960769379
        atten = 60

    '''
    
    # Define p ---------
    if atten < 22:
        p =  0
    elif atten > 121:
        p = 0.5 / ( 1 + ( ( atten - 120 ) / 20 ) ** 5 ) - 2.5 + 0.063 * atten 
    else:
        p = 13 / ( 1 + ( 126 / atten ) ** 1.6 ) - 0.7
        
    # Define n ---------
    if atten > 21 and atten < 121:
        tmp0 = ( 24.3 / ( 1 + ( 149 / atten ) ** 1.6 ) - 0.085 ) / twr * np.pi - 1
        tmp1 = p * np.pi / fcr / 0.95
        n = int( np.round( max( ( tmp0, tmp1 ) ) ) )         
    elif atten > 120 and atten <= 147:
        tmp0 = ( -7.5e-4 * ( atten - 200.3 ) ** 2 + 14.74 ) / twr * np.pi - 1
        tmp1 = p * np.pi / fcr / 0.95
        n = int( np.round( max( ( tmp0, tmp1 ) ) ) )        
    elif atten > 147:
        tmp0 = ( 10.87e-5 * ( atten - 245.6 ) ** 2 -3.1 ) / twr * np.pi - 1
        tmp1 = p * np.pi / fcr / 0.95
        n = int( np.round( max( ( tmp0, tmp1 ) ) ) ) 
        
    alpha = p * np.pi / ( n + 1 ) / fcr   

    return n, alpha, p


def filt_gammatone( fs, freq, bw, n ):

    '''
    
    Impulse response of gammatone filter
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    freq : filter characteristic frequency
    bw : filter bandwidth
    n : filter order
    
    RETURN -------
    
    h : impulse response 
    
    Example: 
        fs = 12500
        freq = 1000
        n = 3
        bw = 36.5605

    Reference:
        Van Immerseel, L., & Peeters, S. (2003). Digital implementation of linear gammatone filters: Comparison of design methods. Acoustics Research Letters Online, 4(3), 59-64.
    
    '''
    
    # Impulse response
    p = 0 # phase
    tvec = np.arange( 0, ( 0.05 * fs ) + 0.05 ) / fs
    h = tvec ** (n-1) * np.exp( -2 * np.pi * bw * tvec ) * np.cos( 2 * np.pi * freq * tvec + p )
    
    # Normalize for unity maximum gain
    h = h / max( abs( np.fft.fft( h, 1024*64 ) ) )
    
    return h


def filt_impulse_response( fcr, n, alpha, p ):

    '''
    
    Low pass filter
    
    INPUT -------
    fcr : frequency cutoff (radians, 0 : pi)
    n : filter length 
    alpha : filter shape parameter
    p : filter smoothing parameter 
    
    RETURN -------
    h : impulse response 
    
    Example: 
        fcr = 0.37699111843077515
        n = 148
        alpha = 0.13082296272948224
        p = 2.3391145736031422

    '''
    
    n_axis = np.arange( -n, n+1 )
    
    h = fcr / np.pi * np.sinc( 1 / np.pi * fcr * n_axis )
    w = np.sinc( 1 / np.pi * alpha * fcr * n_axis / p ) ** p
    h = h * w
    
    return h 
    
    
def filt_lowpass( fs, fc, tw, atten, opt_plot=False ):

    '''
    
    Low pass filter
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    fc : frequency cutoff (Hz)
    tw : transition width (Hz)
    atten : pass attenuation 
    opt_plot : optional plot of filtered power spectrum (true/false)
    
    RETURN -------
    h : impulse response 
    
    Example: 
        fs = 12500
        fc = 750
        tw = 187.5
        atten = 60

    '''
    
    # cutoff frequency and transition witdh in discrete domain 
    fcr = 2 * np.pi * fc / fs 
    twr = 2 * np.pi * tw / fs 
    
    # filter params and window
    n, alpha, p = filt_design( fcr, twr, atten )
    h = filt_impulse_response( fcr, n, alpha, p )
    
    if opt_plot:
        
        m = int( 2 ** np.ceil( np.log2( h.size ) ) * 16 )
        l = int( np.round( m * fc / fs ) )
        hh = abs( np.fft.fft(h,m) )
        ps = 20 * np.log10(hh)
        
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches( 5, 7.5 )
        
        ax[0].plot( np.arange( 1, m+1 )/m*fs, ps, color='k' )
        ax[0].set_xlim( ( 0, fs/2 ) )
        ax[0].set_ylim( ( min( ps[np.isfinite(ps)] ), 0 ) )
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Power spectrum (dB)')
        
        ax[1].plot( np.arange( 1, l+1 )/m*fs, hh[1:l+1], color='k' )
        ax[1].set_xlim( ( 0, l/m*fs ) )
        ax[1].set_ylim( ( 1-10**(-atten/20), 1+10**(-atten/20) ) )
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Linear passband amplitude (dB)') 

    return h
    

def gen_am_sweep_chord( fs, dur, f1, f2, n_carriers, mod_depth, sweep_direction=1 ):
    
    '''
    
    Generate amplitude-modulated sweep with chord carrier (audio waveform). A specific case of gen_sam_chord.
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    f1 : low frequency (Hz)
    f2 : high frequency (Hz), should not exceed fs/2
    n_carriers : number of tone frequency carriers (int)
    mod_depth : modulation depth, from 0 to 1 for unmodulated to max depth 
    sweep_direction: ascending (1) or descending (0)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 0.2
        f1 = 200
        f2 = 3200
        n_carriers = 13 # 2/oct 
        mod_depth = 1
        sweep_direction = 1 
    
    '''
    
    
    mod_freq = 1/dur/2
    
    y = gen_sam_chord( fs, dur, f1, f2, n_carriers, mod_freq, mod_depth )

    if sweep_direction == 0:
        y = np.flip(y)
    
    return y


def gen_am_sweep_noise( fs, dur, mod_depth, sweep_direction=1 ):
    
    '''
    
    Generate amplitude-modulated sweep with white noise carrier (audio waveform). A specific case of gen_sam_noise.
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    mod_depth : modulation depth, from 0 to 1 for unmodulated to max depth 
    sweep_direction: ascending (1) or descending (0)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 0.2
        mod_depth = 1
        sweep_direction = 1 
    
    '''
    
    
    mod_freq = 1/dur/2
    y = gen_sam_noise( fs, dur, mod_freq, mod_depth )
    
    if sweep_direction == 0:
        y = np.flip(y)
    
    return y


def gen_am_sweep_tone( fs, dur, carrier_freq, mod_depth, sweep_direction=1 ):
    
    '''
    
    Generate amplitude-modulated sweep with tone carrier (audio waveform). A specific case of gen_sam_tone.
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    carrier_freq : carrier tone frequency (Hz)
    mod_depth : modulation depth, from 0 to 1 for unmodulated to max depth 
    sweep_direction: ascending (1) or descending (0)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 0.2
        carrier_freq = 1e3 
        mod_depth = 1
        sweep_direction = 1 
    
    '''
    
    
    mod_freq = 1/dur/2
    y = gen_sam_tone( fs, dur, carrier_freq, mod_freq, mod_depth )
    
    if sweep_direction == 0:
        y = np.flip(y)
    
    return y


def gen_binaural_beats( fs, dur, freq_carrier, freq_beat ):
    
    '''
    
    Generate 'binaural beats' (audio waveform)s. 
    Two pure tones separated by a small distance (e.g., 1000 Hz, 1004 Hz) which create a temporal modulation percept when presented binaurally reflecting the frequency difference (e.g., 4 Hz)
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    freq_carrier : frequency of signal (Hz), should not exceed fs/2
    freq_beat : beat frequency (Hz)
  
    RETURN -------
    y : stereo audio signal (sound pressure waveform)

    Example:     
        fs = 96e3
        dur = 5
        freq_carrier = 1e3
        freq_beat = 4.
    
    '''
    

    tvec = np.arange( 0, dur, 1/fs ) # time vector 
    
    y = np.zeros( ( tvec.size, 2 ) )
    
    y[:,0] = np.sin( 2 * np.pi * freq_carrier * tvec )
    y[:,1] = np.sin( 2 * np.pi * (freq_carrier + freq_beat ) * tvec )

    return y


def gen_chord_unif( fs, dur, f1, f2, n_carriers ):
    
    '''
    
    Generate chord with uniform spaced frequencies (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    f1 : low frequency (Hz)
    f2 : high frequency (Hz), should not exceed fs/2
    n_carriers : number of tone frequency carriers (int)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 1
        f1 = 200
        f2 = 3200
        n_carriers = 13 # 2/oct 

    '''

    # Generate chord
    carrier_freqs =  f1 * 2 ** ( np.linspace( 0, np.log2( f2/f1 ), n_carriers ) )
    carriers = np.zeros( ( n_carriers, int(fs*dur) ), dtype=np.float64 )
    for ii in range(n_carriers):
        carriers[ii,:] = np.roll( gen_tone( fs, dur, carrier_freqs[ii] ), np.random.choice( int(fs*dur), 1) ) # roll to minimize destructive/constructive phase interference 
    y = np.sum( carriers, axis=0 ) / n_carriers

    return y
    

def gen_click_train( fs, dur, rate ):
    
    '''
    
    Generate click train (positive square wave pulses) (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    rate : click rate (Hz)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 2
        rate = 4
    
    '''

    stim_samples = int( fs / 2e3 ) # Hard code click pulse duration to 0.002 s
    n_stim = int( np.floor( rate * dur ) )

    y = np.zeros( int( dur*fs ), dtype=np.float64 )
    inter_stim_interval = int( fs / rate )
    
    for ii in range(n_stim):
        idx = int( inter_stim_interval * ii ) 
        y[idx:idx+stim_samples+1] = 1
    
    return y

    
def gen_dynamic_random_chord( fs, dur, f1, f2, pip_dur, pip_atten, pip_density, opt_plot=False ):
    
    '''
        
    Generate dynamic random chord (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    f1 : low frequency (Hz)
    f2 : high frequency (Hz), should not exceed fs/2
    pip_dur : duration of individual tone pips (s)
    pip_atten : attenuation of individual tone pips (dB), may be integer for constant level or list for variable random level within range
    pip_density : pips/oct. Typical values 2-6, must be <= 12
    opt_plot : true/false for stim_matrix plot 

    
    RETURN -------
    y : audio signal (sound pressure waveform)
    stim_matrix : stimulus matrix indicating attenuation levels for each time-frequency bin
    axis_time : time axis for stim matrix (s)
    axis_freq : frequency axis for stim matrix (Hz)
    
    
    Example 1:     
        fs = 48e3
        dur = 3
        f1 = 200.
        f2 = 3200
        pip_dur = 0.02
        pip_atten = 10
        pip_density = 3
        
    Example 2:     
        fs = 48e3
        dur = 3
        f1 = 200.
        f2 = 3200
        pip_dur = 0.05
        pip_atten = [0, 10, 20, 30]
        pip_density = 6
        

    References: 
        DeCharms, R. C., Blake, D. T., & Merzenich, M. M. (1998). Optimizing sound features for cortical neurons. Science, 280(5368), 1439-1444.
        Linden, J. F., Liu, R. C., Sahani, M., Schreiner, C. E., & Merzenich, M. M. (2003). Spectrotemporal structure of receptive fields in areas AI and AAF of mouse auditory cortex. Journal of neurophysiology, 90(4), 2660-2675.
    
    '''

    # Hard code a couple args
    pip_ramp_time = 0.005
    n_bins_oct = 12 # frequency bins per oct

    n_bins_time = int( np.floor( dur / pip_dur ) )  
    n_oct = np.log2( f2/f1 )
    n_bins_freq = int( np.floor( n_oct * n_bins_oct ) ) 
    
    # Store stim values in matrix format
    stim_matrix = np.zeros( ( n_bins_freq, n_bins_time ), dtype=np.float64 )
    stim_matrix[:,:] = -np.inf
    axis_time = np.arange( 0, dur, pip_dur )
    axis_freq = f1 * 2 ** ( np.linspace( 0, np.log2( f2/f1 ), n_bins_freq ) )
    
    y = np.zeros( int(fs*dur), dtype=np.float64 )
    n_pips = int( np.floor( n_oct * pip_density ) )
    n_pip_samples = int( pip_dur * fs )
    
    for ii in range(n_bins_time):
         
        freqs = np.random.choice( n_bins_freq, n_pips, replace=False ) # select frequencies to generate for time step 
        
        y0 = np.zeros( int(fs*pip_dur ), dtype=np.float64 )
        for jj in range(freqs.size):
            
            # Define tone frequency and attenuation 
            freq = axis_freq[ freqs[jj] ]
            if isinstance(pip_atten, int): 
                atten = pip_atten
            elif len( pip_atten ) == 1:
                atten = pip_atten
            else:
                atten = pip_atten[ np.random.choice( len(pip_atten), 1 )[0] ]
              
            # Generate tone and add to chord
            y1 = gen_tone( fs, pip_dur, freq, atten )
            y1 = audio_ramp( y1, fs, pip_ramp_time )            
            y0 += y1
            
            stim_matrix[ freqs[jj], ii ] = atten
            
        y[ n_pip_samples * ii: n_pip_samples * (ii+1) ] = y0 / n_pips
        
    if opt_plot:
        fig, ax = plt.subplots()
        im = ax.imshow( stim_matrix, cmap='RdBu', origin='lower', aspect='auto', extent=[ min(axis_time),max(axis_time), min(axis_freq),max(axis_freq) ]  )
        fig.colorbar(im, ax=ax)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')        
        

        
    return y, stim_matrix, axis_time, axis_freq
        

def gen_dynamic_random_chord_binaural( fs, dur, f1, f2, pip_dur, pip_atten, pip_density, p_left, opt_plot=False ):
    
    '''
        
    Generate dynamic random chord, binaural (audio waveform) 
    Similar to gen_dynamic_random_chord, except with an additional input arg specifying the proportion of tone pips presented through left and right channels. 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    f1 : low frequency (Hz)
    f2 : high frequency (Hz), should not exceed fs/2
    pip_dur : duration of individual tone pips (s)
    pip_atten : attenuation of individual tone pips (dB), may be integer for constant level or list for variable random level within range
    pip_density : pips/oct. Typical values 2-6, must be <= 12
    p_left : proportion tone pips presented through left channel, 1 == all left, 0.5 equal left/right, 0 = all right
    opt_plot : true/false for stim_matrix plot 

    
    RETURN -------
    y : audio signal (sound pressure waveform)
    stim_matrix : stimulus matrix indicating attenuation levels for each time-frequency bin
    axis_time : time axis for stim matrix (s)
    axis_freq : frequency axis for stim matrix (Hz)
    
    
    Example 1:     
        fs = 48e3
        dur = 3
        f1 = 200.
        f2 = 3200
        pip_dur = 0.05
        pip_atten = [0, 10, 20, 30]
        pip_density = 6
        p_left = 0.8
        
    Example 2:     
        fs = 48e3
        dur = 3
        f1 = 200.
        f2 = 3200
        pip_dur = 0.05
        pip_atten = [0, 10, 20, 30]
        pip_density = 2
        p_left = 0.2
        

    References: 
        DeCharms, R. C., Blake, D. T., & Merzenich, M. M. (1998). Optimizing sound features for cortical neurons. Science, 280(5368), 1439-1444.
        Linden, J. F., Liu, R. C., Sahani, M., Schreiner, C. E., & Merzenich, M. M. (2003). Spectrotemporal structure of receptive fields in areas AI and AAF of mouse auditory cortex. Journal of neurophysiology, 90(4), 2660-2675.
    
    '''

    # Hard code a couple args
    pip_ramp_time = 0.005
    n_bins_oct = 12 # frequency bins per oct

    n_bins_time = int( np.floor( dur / pip_dur ) )  
    n_oct = np.log2( f2/f1 )
    n_bins_freq = int( np.floor( n_oct * n_bins_oct ) ) 
    
    # Store stim values in matrix format
    stim_matrix_0 = np.zeros( ( n_bins_freq, n_bins_time ), dtype=np.float64 )
    stim_matrix_0[:,:] = -np.inf
    stim_matrix = np.zeros( ( n_bins_freq, n_bins_time, 2 ), dtype=np.float64 )
    stim_matrix[:,:,:] = -np.inf
    axis_time = np.arange( 0, dur, pip_dur )
    axis_freq = f1 * 2 ** ( np.linspace( 0, np.log2( f2/f1 ), n_bins_freq ) )
    
    y = np.zeros( ( int(fs*dur), 2 ), dtype=np.float64 )
    n_pips = int( np.floor( n_oct * pip_density ) )
    n_pip_samples = int( pip_dur * fs )
    
    # 1/3: populate frequencies for each time bin - - - - - - - - - - - - - - - -
    for ii in range(n_bins_time):
         
        freqs = np.random.choice( n_bins_freq, n_pips, replace=False ) # select frequencies to generate for time step 
        
        for jj in range(freqs.size):
            
            # Define tone frequency and attenuation 
            freq = axis_freq[ freqs[jj] ]
            if isinstance(pip_atten, int): 
                atten = pip_atten
            elif len( pip_atten ) == 1:
                atten = pip_atten
            else:
                atten = pip_atten[ np.random.choice( len(pip_atten), 1 )[0] ]
            
            stim_matrix_0[ freqs[jj], ii ] = atten
            
    # 2/3: randomly assign frequencies to each channel in proportion to p_left arg - - - - - - - - - - - - - - - -
    
    idx_tone = np.nonzero( stim_matrix_0 > -np.inf )
    n_tones = idx_tone[0].size
    idx_l = np.random.choice( n_tones, int( np.ceil( n_tones * p_left ) ), replace=False )
    idx_r = np.setdiff1d( np.arange( 0, n_tones ), idx_l )   
    stim_matrix[ idx_tone[0][idx_l], idx_tone[1][idx_l], 0 ] = stim_matrix_0[ idx_tone[0][idx_l], idx_tone[1][idx_l] ]
    stim_matrix[ idx_tone[0][idx_r], idx_tone[1][idx_r], 1 ] = stim_matrix_0[ idx_tone[0][idx_r], idx_tone[1][idx_r] ]

    # 3/3: generate chords for each channel specified above - - - - - - - - - - - - - - - -
    for ii in range(n_bins_time):
       
        # Left ------
        y0 = np.zeros( int(fs*pip_dur ), dtype=np.float64 )
        idx_tone0 = np.nonzero( stim_matrix[ :, ii, 0 ] > -np.inf )[0]
        if idx_tone0.size > 0: 
            for jj in range(idx_tone0.size):
                
                # Define tone frequency and attenuation 
                freq = axis_freq[ idx_tone0[jj] ]
                atten = stim_matrix[ idx_tone0[jj], ii, 0 ]
                  
                # Generate tone and add to chord
                y1 = gen_tone( fs, pip_dur, freq, atten )
                y1 = audio_ramp( y1, fs, pip_ramp_time )            
                y0 += y1
                
            y0 = y0 / idx_tone0.size
            
        y[ n_pip_samples * ii: n_pip_samples * (ii+1), 0 ] = y0
        
        # Right ------
        y0 = np.zeros( int(fs*pip_dur ), dtype=np.float64 )
        idx_tone0 = np.nonzero( stim_matrix[ :, ii, 1 ] > -np.inf )[0]
        if idx_tone0.size > 0: 
            for jj in range(idx_tone0.size):
                
                # Define tone frequency and attenuation 
                freq = axis_freq[ idx_tone0[jj] ]
                atten = stim_matrix[ idx_tone0[jj], ii, 1 ]
                  
                # Generate tone and add to chord
                y1 = gen_tone( fs, pip_dur, freq, atten )
                y1 = audio_ramp( y1, fs, pip_ramp_time )            
                y0 += y1
                
            y0 = y0 / idx_tone0.size
            
        y[ n_pip_samples * ii: n_pip_samples * (ii+1), 1 ] = y0        

    if opt_plot:
               
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches( 15, 5 )
        
        im = ax[0].imshow( stim_matrix[:,:,0], cmap='RdBu', origin='lower', aspect='auto', extent=[ min(axis_time),max(axis_time), min(axis_freq),max(axis_freq) ]  )
        ax[0].set_xlabel('Time(s)')
        ax[0].set_ylabel('Frequency (Hz)') 
        ax[0].set_title('Left') 

        im = ax[1].imshow( stim_matrix[:,:,1], cmap='RdBu', origin='lower', aspect='auto', extent=[ min(axis_time),max(axis_time), min(axis_freq),max(axis_freq) ]  )
        ax[1].set_xlabel('Time(s)')
        ax[1].set_ylabel('Frequency (Hz)')  
        ax[1].set_title('Right') 
        
        fig.colorbar(im, ax=ax)

    return y, stim_matrix, axis_time, axis_freq
        
               
def gen_fm_sweep( fs, dur, f1, f2, sweep_direction=1 ):
    
    '''
    
    Generate frequency-modulated sweep (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    f1 : low frequency (Hz)
    f2 : high frequency (Hz), should not exceed fs/2
    sweep_direction: ascending (1) or descending (0)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 0.5
        f1 = 200.
        f1 = 2e4
        sweep_direction = 1 
    
    '''

    tvec = np.arange( 0, dur, 1/fs ) # time vector 
    
    # log sweep
    beta = ( np.log( f2/f1 ) ) / dur
    # set zero crossing of sweep at t=0 to match phase with constant freq
    corr_phase_0 = np.remainder( 2 * np.pi * f1 / beta, 2 * np.pi )
    omega_sweep_t = ( 2 * np.pi * f1 ) / beta * np.exp( beta * tvec ) - corr_phase_0
    freq_sweep = np.sin( omega_sweep_t )
    
    if sweep_direction == 0:
        freq_sweep = np.flip(freq_sweep)
    
    y = freq_sweep
    
    return y

def gen_gap_train_noise( fs, dur, gap_dur, rate ):
    
    '''
    
    Generate gap train in noise carrier (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    gap_dur : gap duration (s)
    rate : gap rate (Hz)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 2
        gap_dur = 0.02
        rate = 3
    
    '''

    stim_samples = int( fs * gap_dur ) 
    n_stim = int( np.floor( rate * dur ) )
    
    y = gen_white_noise( fs, dur )
    inter_stim_interval = int( fs / rate )
    
    for ii in range(n_stim):
        idx = int( (inter_stim_interval/2) + ( inter_stim_interval * ii ) ) 
        y[idx:idx+stim_samples+1] = 0
    
    return y


def gen_moving_ripple( fs, dur, f1, f2, mod_spec, mod_temp, contrast_db, n_carriers, downsample_factor, opt_plot=False ):
    
    '''
    
    Generate moving ripple (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    f1 : low frequency (Hz)
    f2 : high frequency (Hz), should not exceed fs/2
    mod_spec: spectral modulation (cycles/oct), must be positive
    mod_temp : temporal modulation (Hz), positive for descending, negative for ascending 
    contrast_db : modulation depth (dB)
    n_carriers : number of tone frequency carriers (int)
    downsample_factor : downsample factor for stim_matrix 
    opt_plot : true/false for stim_matrix plot 
    
    RETURN -------
    y : audio signal (sound pressure waveform)
    stim_matrix : binned time-frequency representation of ripple
    axis_time_df : temporal axis stim_matrix (s)
    axis_oct : frequency axis for stim_matrix (oct)
    
    Example:     
        fs = 96e3  
        dur = 2  
        f1 = 500.
        f2 = 20e3
        mod_spec = 4
        mod_temp = 4
        contrast_db = 40
        n_carriers = 267
        downsample_factor = 22
    
    '''

    n_samples = int( dur * fs )
    ripple_phase = 2 * np.pi * np.random.random_sample()
    oct_max = np.log2(f2/f1)
    axis_oct = np.arange( 0, n_carriers ) /  (n_carriers-1 ) * oct_max 
    axis_freq = f1 * ( 2 ** axis_oct )
    axis_time = np.arange(1,n_samples+1) / fs
    
    y = np.zeros( n_samples )
    stim_matrix = np.zeros( ( n_carriers, int( np.ceil( n_samples/downsample_factor ) ) ) )
    axis_time_df = np.zeros( int( np.ceil( n_samples/downsample_factor ) ) )
    
    for ii in range( n_carriers ):
        
        term_spec  = 2 * np.pi * mod_spec * axis_oct[ii]
        term_temp  = 2 * np.pi * mod_temp * axis_time
        carrier_i = 10 ** ( contrast_db/20 * np.sin( term_spec + term_temp + ripple_phase ) - contrast_db/20 ) # envelope 
        carrier_phase = 2 * np.pi * np.random.random_sample()
        
        y = y + carrier_i * np.sin( 2 * np.pi * axis_freq[ii] * axis_time + carrier_phase ) # carrier 
    
        if ~np.isinf( downsample_factor ): 
            stim_matrix[ii,:] = carrier_i[ np.arange( 0, carrier_i.size, downsample_factor ) ]
            axis_time_df = axis_time[ np.arange( 0, axis_time.size, downsample_factor ) ]
            
    if opt_plot:
        fig, ax = plt.subplots()
        im = ax.imshow( stim_matrix, cmap='RdBu', origin='lower', aspect='auto', extent=[ min(axis_time_df),max(axis_time_df), min(axis_oct),max(axis_oct) ]  )
        fig.colorbar(im, ax=ax)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (oct)')
               
    return y, stim_matrix, axis_time_df, axis_oct
     

def gen_sam_chord( fs, dur, f1, f2, n_carriers, mod_freq, mod_depth ):
    
    '''
    
    Generate sinusoidal amplitude-modulated chord (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    f1 : low frequency (Hz)
    f2 : high frequency (Hz), should not exceed fs/2
    n_carriers : number of tone frequency carriers (int)
    mod_freq : modulation frequency (Hz)
    mod_depth : modulation depth, from 0 to 1 for unmodulated to max depth 

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 1
        f1 = 200
        f2 = 3200
        n_carriers = 13 # 2/oct 
        mod_freq = 4
        mod_depth = 1

    '''

    y = gen_chord_unif( fs, dur, f1, f2, n_carriers )
    
    # Generate modulation envelope 
    # ( Hard code two envelope args )
    energy_constant = False
    phase_start_max = False
    tvec = np.arange( 0, dur, 1/fs ) # time vector 
    if phase_start_max:
        phi = (90/360) * 2 * np.pi # phase of signal (start at max amplitude)
    else:
        phi = (270/360) * 2 * np.pi # phase of signal (start at min amplitude)
    if energy_constant:
        c = ( 1 + (mod_depth**2) / 2 ) ** -0.5  # correction for constant energy (Viemiester's constant)
    else: 
        c = 0.5
    env = ( c * ( 1 + mod_depth * np.sin( ( 2 * np.pi * mod_freq * tvec ) + phi ) ) )
    
    y = y * env
    
    return y


def gen_sam_tone( fs, dur, carrier_freq, mod_freq, mod_depth ):
    
    '''
    
    Generate sinusoidal amplitude-modulated tone (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    carrier_freq : carrier tone frequency (Hz)
    mod_freq : modulation frequency (Hz)
    mod_depth : modulation depth, from 0 to 1 for unmodulated to max depth 

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 1
        carrier_freq = 1e3 
        mod_freq = 4
        mod_depth = 1

    '''
    y = gen_tone( fs, dur, carrier_freq )
    
    # Generate modulation envelope 
    # ( Hard code two envelope args )
    energy_constant = False
    phase_start_max = False
    tvec = np.arange( 0, dur, 1/fs ) # time vector 
    if phase_start_max:
        phi = (90/360) * 2 * np.pi # phase of signal (start at max amplitude)
    else:
        phi = (270/360) * 2 * np.pi # phase of signal (start at min amplitude)
    if energy_constant:
        c = ( 1 + (mod_depth**2) / 2 ) ** -0.5  # correction for constant energy (Viemiester's constant)
    else: 
        c = 0.5
    env = ( c * ( 1 + mod_depth * np.sin( ( 2 * np.pi * mod_freq * tvec ) + phi ) ) )
    
    y = y * env
    
    return y


def gen_sam_noise( fs, dur, mod_freq, mod_depth ):
    
    '''
    
    Generate sinusoidal amplitude-modulated white noise (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    mod_freq : modulation frequency (Hz)
    mod_depth : modulation depth, from 0 to 1 for unmodulated to max depth 

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
    fs = 48e3
    dur = 1
    mod_freq = 4
    mod_depth = 1

    '''

    y = gen_white_noise( fs, dur )

    # Generate modulation envelope 
    # ( Hard code two envelope args )
    energy_constant = False
    phase_start_max = False
    tvec = np.arange( 0, dur, 1/fs ) # time vector 
    if phase_start_max:
        phi = (90/360) * 2 * np.pi # phase of signal (start at max amplitude)
    else:
        phi = (270/360) * 2 * np.pi # phase of signal (start at min amplitude)
    if energy_constant:
        c = ( 1 + (mod_depth**2) / 2 ) ** -0.5  # correction for constant energy (Viemiester's constant)
    else: 
        c = 0.5
    env = ( c * ( 1 + mod_depth * np.sin( ( 2 * np.pi * mod_freq * tvec ) + phi ) ) )

    y = y * env
    
    return y


def gen_sfm_tone( fs, dur, carrier_freq, mod_freq, delta_cams ):
    
    '''
    
    Generate sinusoidal frequency-modulated tone (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    carrier_freq : carrier frequency (Hz)
    mod_freq : modulation frequency (Hz)
    delta_cams : frequency modulation range (cams)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 1
        carrier_freq = 1000
        fm = 4
        delta_cams = 1

    '''

    fm_phase = np.pi
    start_phase = 0 
    
    tvec = np.arange( 0, dur, 1/fs ) # time vector 

    delta_erb = delta_cams * np.cos ( 2 * np.pi * mod_freq * tvec + fm_phase )
    f0 = ( 10 ** (( delta_erb + 21.4 * np.log10( 0.00437 * carrier_freq + 1 )) / 21.4 ) -1 ) / 0.00437
    f_arr = 2 * np.pi * f0 
    ang = ( np.cumsum( f_arr ) / fs ) + start_phase
    
    y = np.sin(ang)

    return y


def gen_tone( fs, dur, freq, atten=0 ):
    
    '''
    
    Generate pure tone (sinusoid) (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    freq : frequency (Hz), should not exceed fs/2
    atten : attenuation (dB)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 0.1
        freq = 2e3
        atten = 10
 
    '''
    
    tvec = np.arange( 0, dur, 1/fs ) # time vector 
    
    y = np.sin( 2 * np.pi * freq * tvec ) * 10**(-atten/20)
    
    return y


def gen_tone_harmonic( fs, dur, freq, n_harmonic ):
    
    '''
    
    Generate harmonic tone (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)
    freq : frequency (Hz), should not exceed fs/2
    n_harmonic : number of harmonics

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    Example: 
        fs = 48e3
        dur = 0.5
        freq = 500
        n_harmonic = 4
 
    '''
    
    # Fundamental
    y = gen_tone( fs, dur, freq ) 
    
    # Harmonics
    freq0 = freq
    for ii in range( n_harmonic ):
        freq0 *= 2
        y += np.roll( gen_tone( fs, dur, freq0 ), np.random.choice( int(fs*dur), 1) ) # roll to minimize destructive/constructive phase interference 
    y /= ( n_harmonic + 1 )       

    return y


def gen_white_noise( fs, dur ):
    
    '''
    
    Generate white noise (audio waveform) 
    
    INPUT -------
    fs : audio sample rate, e.g., 48e3
    dur : duration (s)

    RETURN -------
    y : audio signal (sound pressure waveform)
    
    '''
    
    y = ( np.random.rand( int( np.round(dur * fs) ) ) * 2 ) - 1
    
    return y


def get_cochleagram( y, fs, f1, f2, dx, fm_max, of, atten=60, opt_norm='amp', opt_filt='gamma', opt_bw='erb', opt_plot=False ):

    '''

    Cochelagram model of audio waveform.
    
    INPUT -------
    y : audio signal (sound pressure waveform)
    fs : audio sample rate (Hz), e.g., 48e3
    f1 : low frequency (Hz)
    f2 : high frequency (Hz)
    dx : spectral separation betwen adjacent filters in octaves, e.g., 1/8
    fm_max : frequency-modulation max allowed for temporal envelope at each band. In inf, full range is used.
    of: oversampling factor for temporal envelope. Since the max frequency of the envelope is fm_max, the frequency used to sample the envelope is 2*fm_max*of
    atten : attenuation error for modulation low pass filter, dB
    opt_norm : type of normalization, options == 'amp' for equal amplitude, 'en' for equal energy
    opt_filt : type of filter, options == 'gamma', 'bspline'
    opt_bw : type of bandwidth, options == 'erb' for equivalent rectangular bandwidth, 'cb' for critical bandwidth 
    opt_plot : optional cochleagram plot (true/false)

    RETURN -------
    c : cochleagram 
    cc : cochleagram corrected for group delay 
    c_db : cochleagram in dB units
    cc_db : cochleagram corrected for group delay in dB units
    axis_time : time axis for cochleagram (s)
    axis_freq : frequency axis for stim matrix (Hz)
      
    Example: 
        f1 = 100
        f2 = fs/2
        dx = 0.1
        fm_max = 750
        of = 2
        atten = 60
        opt_norm ='amp'
        opt_filt = 'gamma'
        opt_bw = 'erb'    
        
    Attribution: 
        This function is a revised implementation of MATLAB functions provided by:
        Khatami, F., & Escabí, M. A. (2020). Spiking network optimized for word recognition in noise predicts auditory system hierarchy. PLoS computational biology, 16(6), e1007558.
    
    '''
    
    # hard coded params - - - - 
    n = 3 # filter order
    delay = 0
    opt_truncate = True
        
    # Frequency axis for chromatically-spaced filter bank (f[k] = f[k+1]*2**dx)
    xn = np.log2( f2/f1 )
    nc = np.floor( xn/dx )
    xc = np.arange( 0.5, nc+0.5 ) / nc * xn
    axis_freq = f1 * 2 ** xc
    
    if opt_bw == 'erb':
        bw = get_erb( axis_freq )
    elif opt_bw == 'cb':
        bw = get_critical_bw( axis_freq )
    
    # Temporal downsampling factor 
    df = int( max( ( 1, np.ceil( fs / 2 / fm_max / of ) ) ) )
    dfi = np.arange( 0, y.size, df ) # indices
    
    # Low pass filter for extracting envelope 
    fc = fm_max
    tw = fc * 0.25
    h_lo = filt_lowpass( fs, fc, tw, atten )
    n_lo = int( ( h_lo.size - 1 ) / 2 )
    
    # Generate filters 
    if opt_filt == 'gamma':
        htmp = filt_gammatone( fs, axis_freq[0], bw[0], n )
        filters = np.zeros( ( axis_freq.size, htmp.size ), dtype=float )
        filter_n = np.zeros( axis_freq.size, dtype=float )
        for ii in range(axis_freq.size):
            filters[ii,:] = filt_gammatone( fs, axis_freq[ii], bw[ii], n )
            filter_n[ii] = ( filters[ii,:].size - 1 ) / 2   
            
    # Find group delays 
    group_delay = np.zeros( axis_freq.size, dtype=float )
    for ii in range(axis_freq.size):
        p = filters[ii,:] ** 2 / np.sum( filters[ii,:] ** 2 )
        t = np.arange( 1, filters[ii,:].size+1 ) / fs
        group_delay[ii] = np.sum( p * t )
    nmax = round( max( group_delay * fs ) )
    ndelay = round( group_delay[0] * fs ) +1
    dfic = np.arange( ndelay, y.size - nmax + ndelay -1 +df, df ) # delay corrected downsampled indices
    
    # Filter signal, extract envelope, downsample, store cochleagram model
    nfft = int( 2 ** nextpow2( y.size + max(filter_n)*2+1 + n_lo*2+1 ) ) # FFT size
    c = np.zeros( ( axis_freq.size, dfi.size ), dtype=float )
    cc = np.zeros( ( axis_freq.size, dfic.size ), dtype=float )
    for ii in range(axis_freq.size):
        
        # Filter (gammatone or bspline)
        h = filters[ii,:]
        if opt_norm == 'en':
            h = h / np.sqrt( np.sum( h ** 2 ) ) # equal energy 
        Y = filt_convfft( y, h, delay, nfft, opt_truncate )
        
        # Low pass filter 
        Y = abs( signal.hilbert( Y ) ) # Get envelope via Hilbert transform
        Y = filt_convfft( Y, h_lo, n_lo )
        Y[Y<0] = 0 # remove negative values due to filtering
        
        # store cochleagram frequency band
        c[ii,:] = Y[dfi] # downsample
    
        # ...(delay corrected)  
        ndelay = round( group_delay[0] * fs ) +1
        cc[ii,:] = Y[ dfic ] # downsample
    
    # dB models - - - - - 
    c_db = 20 * np.log10(c)
    idx = np.isfinite( c_db )
    mn = min( c_db[idx] )
    idx = np.isinf( c_db )
    c_db[idx] = mn # replace -inf values with min
    c_db -= np.mean( c_db ) # mean subtraction 
    
    cc_db = 20 * np.log10(cc)
    idx = np.isfinite( cc_db )
    mn = min( cc_db[idx] )
    idx = np.isinf( cc_db )
    cc_db[idx] = mn # replace -inf values with min
    cc_db -= np.mean( cc_db ) # mean subtraction 
    
    axis_time = np.arange( 0, c.shape[1] ) / (fs/df)
    
    if opt_plot:

        plt.close('all')
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches((18,12))
                
        axs[0,0].imshow( c, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(axis_time), max(axis_time), min(axis_freq), max(axis_freq) ] )
        axs[0,0].set_xlabel('Time (s)')
        axs[0,0].set_ylabel('Frequency (Hz)')
        axs[0,0].set_title('Cochleagram')
        
        axs[0,1].imshow( cc, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(axis_time), max(axis_time), min(axis_freq), max(axis_freq) ] )
        axs[0,1].set_xlabel('Time (s)')
        axs[0,1].set_ylabel('Frequency (Hz)')
        axs[0,1].set_title('Cochleagram - delay corrected')
        
        axs[1,0].imshow( c_db, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(axis_time), max(axis_time), min(axis_freq), max(axis_freq) ] )
        axs[1,0].set_xlabel('Time (s)')
        axs[1,0].set_ylabel('Frequency (Hz)')
        axs[1,0].set_title('Cochleagram dB')
        
        axs[1,1].imshow( cc_db, cmap='gray_r', origin='lower', aspect='auto', extent=[ min(axis_time), max(axis_time), min(axis_freq), max(axis_freq) ] )        
        axs[1,1].set_xlabel('Time (s)')
        axs[1,1].set_ylabel('Frequency (Hz)')
        axs[1,1].set_title('Cochleagram dB - delay corrected')        

    return c, cc, c_db, cc_db, axis_time, axis_freq


def get_critical_bw( f ):
    
    '''
    
    Calc critical bandwidth  
    
    INPUT -------
    f : frequency (Hz)

    RETURN -------
    b :  critical bandwidth
    
    Example: 
        f = 1e3
        
    '''
    
    b = 94 + 71 * ( f / 1000 ) ** (3/2)
        
    return b
    

def get_erb( f ):
    
    '''
    
    Calc equivalent rectangular bandwidth  
    
    INPUT -------
    f : frequency (Hz)

    RETURN -------
    b :  equivalent rectangular bandwidth
    
    Example: 
        f = 1e3
        
    '''
    
    b = 1.019 * 24.7 * ( 1 + 4.37 * f / 1000 )
        
    return b


def nextpow2( x ):
    
    '''
    
    Next higher power of 2, the first p such that 2.^p >= abs(x)

    INPUT -------
    x : integer

    RETURN -------
    n : power
    
    '''
    
    p = 2
    while p < x: 
        p = p * 2
    p = np.log2( p )
    
    
    return p


def plot_spectrogram( y, fs ):
    
    '''
    
    Plot spectrogram, time-frequency representation of sound 
    
    INPUT -------
    y : audio signal (sound pressure waveform)
    fs : audio sample rate, e.g., 48e3
    
    '''
    
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches( 5, 10 )

    # Plot sound pressure waveform - - - - - - - - -
    f, t, Sxx = signal.spectrogram( y, fs )
    ax[0].plot( y )
    ax[0].set_xlabel('Time (Sample)')
    ax[0].set_ylabel('Amplitude (AU)') 
        
    # Plot spectrogram - - - - - - - - - 
    ax[1].pcolormesh( t, f, Sxx, shading='auto' )
    ax[1].set_xlabel('Time(s)')
    ax[1].set_ylabel('Frequency (Hz)')   
    
