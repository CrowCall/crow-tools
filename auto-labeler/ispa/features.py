from collections import defaultdict
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torchaudio.models import wav2vec2_model


class AvesFeatureExtractor(nn.Module):

    # default config and model paths
    config_path = 'models/aves-base-bio.torchaudio.model_config.json'
    model_path = 'models/aves-base-bio.torchaudio.pt'

    def __init__(self, aves_config_path=config_path, aves_model_path=model_path):

        super().__init__()

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html

        self.config = self.load_config(aves_config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(aves_model_path))
        self.model.feature_extractor.requires_grad_(False)
        self.model.eval()

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        with torch.no_grad():
            # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
            out = self.model.extract_features(sig)[0][-1]

        return out


class MFCCFeatureExtractor(nn.Module):

    def __init__(self, sr=16000, n_mfcc=40, n_fft=400, hop_length=320, power=2.0, **kwargs):
        super().__init__()

        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=self.n_mfcc,
            melkwargs={'n_fft': self.n_fft, 'hop_length': self.hop_length, 'n_mels': self.n_mfcc},
        )

    def forward(self, sig):
        out = self.mfcc(sig)
        out = out.transpose(1, 2)
        return out


def enumerate_tokens(features, start, distance):
    # enumerate list of tokens that start at start index
    # start: start index in features
    results = []
    # 1 frame of MFCC/AVES = 20ms
    for length in [1, 2, 5, 10, 20, 50]:    # multiple of 20ms
        end = start + length
        if end > len(features):
            break
        # choose the cluster with the smallest distance
        c_min = distance[start:end, :].mean(axis=0).argmin()
        dist = distance[start:end, c_min].sum()

        results.append({
            'type': 'note',
            'length': length,
            'cluster': c_min,
            'dist': dist})
        # print(f"note: {start} {length} {c_min} {dist}")

    return results


_BOS_TOKEN = {'type': 'BOS', 'length': 1, 'cluster': None, 'dist': 0.}
_EOS_TOKEN = {'type': 'EOS', 'length': 1, 'cluster': None, 'dist': 0.}


def cost_func(cnode, bnode=None):
    distance_cost = cnode['token']['dist']
    length_cost = 10. / (1 + np.log10(cnode['token']['length']))
    total_cost = distance_cost + length_cost

    # print(f"cost: {cnode['start']} {cnode['token']['length']} {cnode['token']['dist']} {distance_cost} {length_cost} {total_cost} {total_cost / cnode['token']['length']}")
    return total_cost


def run_viterbi(features, kmeans, cost_func=lambda node: 0):
    # features: (time, feature)
    distance = kmeans.transform(features)   # (time, n_clusters)
    bos_node = {'start': -1, 'next': [], 'token': _BOS_TOKEN, 'cost': 0}
    end_node_list = defaultdict(list)
    end_node_list[0].append(bos_node)

    for i in range(0, len(features)+1):
        # print(f"i: {i}", file=sys.stderr)
        if i < len(features):
            tokens = enumerate_tokens(features, i, distance)
        else:
            tokens = [_EOS_TOKEN]

        for token in tokens:
            cnode = {'start': i, 'next': [], 'token': token}
            min_cost = -1
            min_bnodes = []

            for bnode in end_node_list[i]:
                cost = bnode['cost'] + cost_func(cnode, bnode)

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

LEN2MARK = {
    1: '',
    2: ',',
    5: ':',
    10: ';',
    20: '.',
    50: '..',
}

def convert_to_text(tokens, phoneme_map=None):
    result = []
    for token in tokens:
        if token['type'] == 'note':
            if phoneme_map is not None:
                token_str = phoneme_map[str(token['cluster'])].lower()
            else:
                token_str = str(token['cluster'])
            result.append(f"{token_str}{LEN2MARK[token['length']]}")
        elif token['type'] == 'BOS':
            pass
        elif token['type'] == 'EOS':
            pass
        else:
            raise ValueError(f'Unknown token type: {token["type"]}')

    return ' '.join(result)


class FeatureBasedISPAPredictor:
    def __init__(self, feature_type=None, kmeans_model=None, phoneme_map=None, use_cuda=False, **kwargs):
        if feature_type == 'aves':
            self.feature_extractor = AvesFeatureExtractor(**kwargs)
        elif feature_type == 'mfcc':
            self.feature_extractor = MFCCFeatureExtractor()
        else:
            raise ValueError('feature_type must be "aves" or "mfcc"')
        pass

        with open(kmeans_model, 'rb') as f:
            self.kmeans = pickle.load(f)

        if phoneme_map is not None:
            with open(phoneme_map, 'r') as f:
                self.phoneme_map = json.load(f)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.feature_extractor = self.feature_extractor.cuda()

    def predict(self, waveform, variation=None):
        if variation not in {'raw', 'seg', 'phn'}:
            raise ValueError('variation must be "raw", "seg", or "phn"')

        # extract feature
        if self.use_cuda:
            waveform = waveform.cuda()
        feature = self.feature_extractor(waveform)   # (batch, time, feature)
        feature = feature.squeeze(0)            # (time, feature)

        # apply k-means
        if self.use_cuda:
            feature = feature.cpu()
        feature = feature.detach().numpy()

        if variation == 'raw':
            clusters = self.kmeans.predict(feature)
            text = ' '.join([str(x) for x in clusters])
        elif variation == 'seg':
            tokens = run_viterbi(feature, self.kmeans, cost_func=cost_func)
            text = convert_to_text(tokens)
        elif variation == 'phn':
            tokens = run_viterbi(feature, self.kmeans, cost_func=cost_func)
            text = convert_to_text(tokens, self.phoneme_map)

        return text
