from collections import defaultdict
from dataclasses import dataclass
import re

import librosa
import numpy as np
import pesto
import pandas as pd


def length_to_name(length):
    # convert length (multiples of 1/32 seconds) to note name
    # assuming bpm = 60
    return {
        1: '/32',
        2: '/16',
        4: '/8',
        8: '/4',
        16: '/2',
        32: '',      # quarter note
        64: 'x2',
        128: 'x4',
    }[length]


def bw_to_name(bw):
    # convert relative bandwidth to name
    if bw < 0.4:
        return 'U'
    elif bw < 0.8:
        return 'N'
    elif bw < 1.2:
        return 'M'
    elif bw < 1.6:
        return 'W'
    else:
        return 'X'


def estimate_pitch_pesto(waveform, sr):
    time, frequency, confidence, _ = pesto.predict(
        waveform, sr,
        step_size=31.25,
        reduction='alwa',
        convert_to_freq=True)

    # convert to df
    df = pd.DataFrame({
        'time': time,
        'frequency': frequency,
        'confidence': confidence})

    return df


def get_amplitude(df, waveform, sr):
    # get amplitude of waveform per frame
    frame_length = int(0.03125 * sr)      # 1/32 sec per frame
    amps = []
    for i in range(0, len(df)):
        st = int(df['time'][i] * sr)
        ed = st + frame_length
        # shift st and ed to the center of the frame
        st = max(0, st - frame_length // 2)
        ed = min(len(waveform[0]), ed + frame_length // 2)
        amp = np.mean(np.abs(waveform[0][st:ed].numpy()))
        # convert to dB
        amp = 20 * np.log10(amp + 1e-3)
        amps.append(amp)
    df['amplitude'] = amps

    return df


def get_relative_bandwidth(df, waveform, sr):
    frame_length = int(0.03125 * sr)      # 1/32 sec per frame
    bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform[0].numpy(),
        n_fft=frame_length,
        hop_length=frame_length)
    centroid = librosa.feature.spectral_centroid(
        y=waveform[0].numpy(),
        n_fft=frame_length,
        hop_length=frame_length)

    centroid += 1e-3
    df['bandwidth'] = (bandwidth / centroid)[0]
    return df


def get_residual(x, y, freq, slope):
    if slope.value < 0:
        start_freq = freq * -slope.value
        end_freq = freq - (start_freq - freq)
    else:
        end_freq = freq * slope.value
        start_freq = freq - (end_freq - freq)
    # compute residual as sum of absolute error
    if x[-1] == x[0]:
        residual = 0.
    else:
        residual = np.sum(np.abs(y - (start_freq + (end_freq - start_freq) * x / (x[-1] - x[0]))))

    return residual


# dataclass for slope
@dataclass
class Slope:
    name: str
    value: float

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


SLOPE_FLAT = Slope("=", 1)
SLOPES = [
    Slope("-3", -2),
    Slope("-2", -1.5),
    Slope("-1", -1.2),
    SLOPE_FLAT,
    Slope("+1", 1.2),
    Slope("+2", 1.5),
    Slope("+3", 2)
]


def enumerate_tokens(df, start):
    # enumerate list of tokens that start at start index
    # start: start index in df
    results = []
    for length in [1, 2, 4, 8, 16, 32, 64, 128]:    # multiple of 1/32 sec
        end = start + length
        if end > len(df):
            break
        if df['amplitude'][start:end].max() < -50:
            # rest
            results.append({
                'type': 'rest',
                'length': length,
                'freq': None,
                'res': None,
                'conf': None,
                'amp': None,
                'bw': None})
        else:
            if length == 1:
                slopes = [SLOPE_FLAT]
            else:
                slopes = SLOPES

            df_slice = df.iloc[start:end]
            x_slice = df_slice['time'].values
            x_slice = x_slice - x_slice[0]
            y_slice = df_slice['frequency'].values
            freq_slice = np.mean(y_slice)

            conf = np.mean(df_slice['confidence'].values)
            amp = np.mean(df_slice['amplitude'].values)
            bw = np.mean(df_slice['bandwidth'].values)

            for slope in slopes:
                # res = fit_sine_wave_with_fixed_slope(df, start, end, slope)
                # freq, res, conf, amp, bw = res

                res = get_residual(x_slice, y_slice, freq_slice, slope)
                freq = freq_slice

                results.append({
                    'type': 'note',
                    'length': length,
                    'freq': freq,
                    'slope': slope,
                    'res': res,
                    'conf': conf,
                    'amp': amp,
                    'bw': bw})

    return results


_BOS_TOKEN = {'type': 'BOS', 'length': 1, 'freq': None, 'res': None, 'conf': None, 'amp': None}
_EOS_TOKEN = {'type': 'EOS', 'length': 1, 'freq': None, 'res': None, 'conf': None, 'amp': None}


def cost_func(cnode, bnode=None):
    if cnode['token']['type'] == 'note':
        residual_cost = cnode['token']['res']
        conf_cost = 100 * (1 - cnode['token']['conf'])

    else:
        residual_cost = 0
        conf_cost = 0

    length_cost = 1_000 / (1 + np.log2(cnode['token']['length']))
    total_cost = residual_cost + length_cost

    return total_cost


def run_viterbi(df, cost_func=lambda node: 0):
    bos_node = {'start': -1, 'next': [], 'token': _BOS_TOKEN, 'cost': 0}
    end_node_list = defaultdict(list)
    end_node_list[0].append(bos_node)

    for i in range(0, len(df)+1):
        if i < len(df):
            tokens = enumerate_tokens(df, i)
        else:
            tokens = [_EOS_TOKEN]
        for token in tokens:
            cnode = {'start': i, 'next': [], 'token': token}
            min_cost = -1
            min_bnodes = []

            cnode_cost = cost_func(cnode)
            for bnode in end_node_list[i]:
                cost = bnode['cost'] + cnode_cost

                # allow ties
                # if min_cost == -1 or cost < min_cost:
                #     min_cost = cost
                #     min_bnodes = [bnode]
                # elif cost == min_cost:
                #     min_bnodes.append(bnode)

                # doesn't allow ties
                if min_cost == -1 or cost < min_cost:
                    min_cost = cost
                    min_bnodes = [bnode]

            if len(min_bnodes) > 0:
                for bnode in min_bnodes:
                    cnode['cost'] = min_cost
                    bnode['next'].append(cnode)

                end_nodes = end_node_list[i+token['length']]
                if not cnode in end_nodes:
                    end_nodes.append(cnode)

    solutions_cache = {}

    def _enum_solutions(node):
        # convert node to hashable
        cache_key = (node['start'], tuple(node['token'].items()))
        if cache_key in solutions_cache:
            return solutions_cache[cache_key]

        if node['token'] == _EOS_TOKEN:
            return [[_EOS_TOKEN]]

        solutions = []
        for next_node in node['next']:
            for solution in _enum_solutions(next_node):
                node['token']['start'] = node['start']
                solutions.append([node['token']] + solution)

        solutions_cache[cache_key] = solutions
        return solutions

    return _enum_solutions(bos_node)[0]


def convert_to_text(tokens):
    result = []
    prev_end = 0
    for token in tokens:
        if token['type'] == 'note':
            if token['start'] > prev_end:
                length_name = length_to_name(token['start']-prev_end)
                result.append(f"R{length_name}")
            bw_name = bw_to_name(token['bw'])
            note_name = librosa.hz_to_note(token['freq'], unicode=False)
            # extract just the octave number with regex
            note_name = re.sub(r'[A-G]#?', '', note_name)

            length_name = length_to_name(token['length'])
            slope_name = token['slope'].name
            result.append(f"{bw_name}{note_name}{length_name}{slope_name}")
            prev_end = token['start'] + token['length']
        elif token['type'] == 'BOS':
            pass
        elif token['type'] == 'rest':
            result.append(f"R{length_to_name(token['length'])}")
            prev_end = token['start'] + token['length']

    return ' '.join(result)


def run_inference(waveform, sr):
    df = estimate_pitch_pesto(waveform, sr)
    df = get_amplitude(df, waveform, sr)
    df = get_relative_bandwidth(df, waveform, sr)

    solution = run_viterbi(df, cost_func=cost_func)

    text = convert_to_text(solution)
    ispa_results = {
        'df': df,
        'solution': solution,
        'text': text
    }
    return ispa_results
