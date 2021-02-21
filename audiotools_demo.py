
# -*- coding: utf-8 -*-
"""

Demo of various audiotools functions.

@author: jamesbigelow at gmail dot com

"""

#### Import modules 
import audiotools as at
import sounddevice as sd


#%% Click train 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 0.5 # duration in seconds 
rate = 4 # click rate 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 

# Generate signal and attenuate - - - - - - - - -
y = at.gen_click_train( fs, dur, rate )
y = at.audio_attenuate( y, atten ) 

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'click_train_demo.wav'
at.audio_write_wav( y, fs, fid )

#%% Gap train - noise carrier 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 5 # duration in seconds 
gap_dur = 0.005
rate = 2 # click rate 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds 

# Generate signal, attenuate, and ramp - - - - - - - - -
y = at.gen_gap_train_noise( fs, dur, gap_dur, rate )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'gap_train_noise_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% Amplitude-modulated sweep - noise carrier 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 1 # duration in seconds 
mod_depth = 1 # modulation depth, from 0 to 1 for unmodulated to max depth        
sweep_direction = 1 # 1 == ascending, 0 == descending 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds 

# Generate signal, attenuate, and ramp - - - - - - - - -
y = at.gen_am_sweep_noise( fs, dur, mod_depth, sweep_direction )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'am_sweep_noise_demo.wav'
at.audio_write_wav( y, fs, fid )

#%% Amplitude-modulated sweep - tone carrier 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 0.2 # duration in seconds 
carrier_freq = 1e3 
mod_depth = 1 # modulation depth, from 0 to 1 for unmodulated to max depth        
sweep_direction = 1 # 1 == ascending, 0 == descending 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds 

# Generate signal, attenuate, and ramp - - - - - - - - -
y = at.gen_am_sweep_tone( fs, dur, carrier_freq, mod_depth, sweep_direction )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'am_sweep_tone_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% Frequency-modulated sweep 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 0.5 # duration in seconds 
f1 = 200. # low freq
f2 = 10000. # high freq
sweep_direction = 0 # 1 == ascending, 0 == descending 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds 

# Generate signal, attenuate, and ramp - - - - - - - - -
y = at.gen_fm_sweep( fs, dur, f1, f2, sweep_direction )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'fm_sweep_demo.wav'
at.audio_write_wav( y, fs, fid )

#%% Sinusoidal amplitude-modulated chord

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 1 # duration in seconds 
f1 = 200 # low freq
f2 = 3200 # high freq
n_carriers = 13 # 2/oct 
mod_freq = 4 # modulation frequency
mod_depth = 1 # modulation depth, from 0 to 1 for unmodulated to max depth        
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 

# Generate signal and attenuate - - - - - - - - -
y = at.gen_sam_chord( fs, dur, f1, f2, n_carriers, mod_freq, mod_depth )
y = at.audio_attenuate( y, atten ) 

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'sam_chord_demo.wav'
at.audio_write_wav( y, fs, fid )

#%% Sinusoidal amplitude-modulated noise

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 1 # duration in seconds 
mod_freq = 4 # modulation frequency
mod_depth = 1 # modulation depth, from 0 to 1 for unmodulated to max depth        
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 

# Generate signal and attenuate - - - - - - - - -
y = at.gen_sam_noise( fs, dur, mod_freq, mod_depth )
y = at.audio_attenuate( y, atten ) 

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'sam_noise_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% Sinusoidal amplitude-modulated tone

# Define stim tone params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 1 # duration in seconds 
carrier_freq = 1e3 # tone carrier frequency 
mod_freq = 4 # modulation frequency
mod_depth = 1 # modulation depth, from 0 to 1 for unmodulated to max depth        
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 

# Generate signal and attenuate - - - - - - - - -
y = at.gen_sam_tone( fs, dur, carrier_freq, mod_freq, mod_depth )
y = at.audio_attenuate( y, atten ) 

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'sam_tone_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% Sinusoidal frequency-modulated tone

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 1 # duration in seconds 
carrier_freq = 4000 # carrier frequency
mod_freq = 4 # modulation frequency
delta_cams = 4 # modulation range

# Generate signal and attenuate - - - - - - - - -
y = at.gen_sfm_tone( fs, dur, carrier_freq, mod_freq, delta_cams )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'sfm_tone_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% Dynamic random chord 

# Define stim params - - - - - - - - -
fs = 48e3
dur = 3
f1 = 200.
f2 = 3200
pip_dur = 0.05
pip_atten = [0, 10, 20, 30]
pip_density = 6
opt_plot = True 
        
# Generate signal - - - - - - - - -
y, stim_matrix, axis_time, axis_freq = at.gen_dynamic_random_chord( fs, dur, f1, f2, pip_dur, pip_atten, pip_density, opt_plot )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'drc_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% Dynamic random chord - binaural 

# Define stim params - - - - - - - - -
fs = 48e3
dur = 3
f1 = 200.
f2 = 3200
pip_dur = 0.05
pip_atten = [0, 10, 20, 30]
pip_density = 6
p_left = 1
opt_plot = True 
        
# Generate signal - - - - - - - - -
y, stim_matrix, axis_time, axis_freq = at.gen_dynamic_random_chord_binaural( fs, dur, f1, f2, pip_dur, pip_atten, pip_density, p_left, opt_plot )

# Plot - - - - - - - - - 
at.plot_spectrogram( y[:,0], fs )
at.plot_spectrogram( y[:,1], fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'drc_binaural_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% Moving ripple 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 2 # duration in seconds 
f1 = 200. # low freq
f2 = 10000. # high freq
mod_spec = 4
mod_temp = -4
contrast_db = 40
n_carriers = 267
downsample_factor = 22
opt_plot = True 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds 

# Generate signal, attenuate, and ramp - - - - - - - - -
y, stim_matrix, axis_time_df, axis_oct = at.gen_moving_ripple( fs, dur, f1, f2, mod_spec, mod_temp, contrast_db, n_carriers, downsample_factor, opt_plot )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'moving_ripple_demo.wav'
at.audio_write_wav( y, fs, fid )

#%% Tone 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 0.5 # duration in seconds 
freq = 2e3 # frequency (Hz)
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds

# Generate signal and ramp - - - - - - - - -
y = at.gen_tone( fs, dur, freq, atten )
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'tone_demo.wav'
at.audio_write_wav( y, fs, fid )

#%% Harmonic tone 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 0.5 # duration in seconds 
freq = 500 # frequency (Hz)
n_harmonic = 6 # number of harmonics 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds

# Generate signal, attenuate, and ramp - - - - - - - - -
y = at.gen_tone_harmonic( fs, dur, freq, n_harmonic )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'harmonic_tone_demo.wav'
at.audio_write_wav( y, fs, fid )


#%% White noise 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 2 # duration in seconds 
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds

# Generate signal, attenuate, and ramp - - - - - - - - -
y = at.gen_white_noise( fs, dur )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y, fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'white_noise_demo.wav'
at.audio_write_wav( y, fs, fid )
              

#%% Binaural beats 

# Define stim params - - - - - - - - -
fs = 48e3 # audio sample rate 
dur = 5 # duration in seconds 
freq_carrier = 500.
freq_beat = 8
atten = 10 # sound attenuation in dB, 0 == unattenuated (loudest possible) 
ramp_time = 0.020 # ramp duration in seconds

# Generate signal, attenuate, and ramp - - - - - - - - -
y = at.gen_binaural_beats( fs, dur, freq_carrier, freq_beat )
y = at.audio_attenuate( y, atten ) 
y = at.audio_ramp( y, fs, ramp_time )

# Plot - - - - - - - - - 
at.plot_spectrogram( y[:,0], fs )
at.plot_spectrogram( y[:,1], fs )

# Play audio - - - - - - - - -
sd.play( y, fs ) 
sd.wait()

# Write to .wav file - - - - - - - - -
fid = 'binaural_beats_demo.wav'
at.audio_write_wav( y, fs, fid )

