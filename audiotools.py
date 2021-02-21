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
    
    Atenuate (audio waveform) 
    
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
    
