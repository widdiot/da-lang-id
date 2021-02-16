import os
import sys
import yaml
import random
import save_segs
from kaldi.feat.mfcc import MfccOptions, Mfcc
from kaldi.feat.window import FrameExtractionOptions
from kaldi.feat.mel import MelBanksOptions
from kaldi.matrix import Vector, SubVector, DoubleVector, Matrix
from kaldi.util.table import MatrixWriter, SequentialMatrixReader, RandomAccessPosteriorReader, SequentialWaveReader, \
    DoubleVectorWriter, SequentialVectorReader
from kaldi.ivector import IvectorExtractor, IvectorExtractorUtteranceStats, compute_vad_energy, VadEnergyOptions, \
    LogisticRegressionConfig, LogisticRegression
from kaldi.feat.functions import sliding_window_cmn, SlidingWindowCmnOptions
from kaldi.ivector import LogisticRegressionConfig, LogisticRegression
from kaldi.feat.functions import sliding_window_cmn, SlidingWindowCmnOptions
from kaldi.matrix import Vector, SubVector, DoubleVector, Matrix
from kaldi.util.io import xopen
from kaldi.matrix.common import MatrixTransposeType
from nn_speech_models import *
import kaldiio
import numpy as np
import torch

if len(sys.argv) != 2:
    sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_pah = sys.argv[1]  # '/LANG-ID-X/config_1.yml'

config_args = yaml.safe_load(open(config_file_pah))

############## idx2label #################

i2l = {0: 'bengali', 1: 'english', 2: 'hindi', 3: 'kannada', 4: 'malayalam', 5: 'marathi', 6: 'tamil', 7: 'telugu'}

############## model #################

nn_LID_model_DA = ConvNet_LID_DA(
    feature_dim=config_args['model_arch']['feature_dim'],
    bottleneck=config_args['model_arch']['bottleneck'],
    bottleneck_size=config_args['model_arch']['bottleneck_size'],
    output_dim=config_args['model_arch']['output_dim'],
    dropout_frames=config_args['model_arch']['frame_dropout'],
    dropout_features=config_args['model_arch']['feature_dropout'],
    signal_dropout_prob=config_args['model_arch']['signal_dropout_prob'],
    num_channels=config_args['model_arch']['num_channels'],
    num_classes=len(i2l),  # or config_args['model_arch']['num_classes'],
    filter_sizes=config_args['model_arch']['filter_sizes'],
    stride_steps=config_args['model_arch']['stride_steps'],
    pooling_type=config_args['model_arch']['pooling_type'])

nn_LID_model_DA.load_state_dict(torch.load(config_args['best_model'], map_location=torch.device('cpu')))

print(nn_LID_model_DA)
############## cmn #################

cmvn_stats = kaldiio.load_mat(config_args['source_cmvn'])
mean_stats = cmvn_stats[0,:-1]
count = cmvn_stats[0,-1]
offset = np.expand_dims(mean_stats,0)/count
CMVN = offset

############## Mfcc opts #################
fopts = FrameExtractionOptions()
fopts.samp_freq = 16000
fopts.snip_edges = True

hires_mb_opts = MelBanksOptions()
hires_mb_opts.low_freq = 40
hires_mb_opts.high_freq = -200
hires_mb_opts.num_bins = 40
hires_mfcc_opts = MfccOptions()
hires_mfcc_opts.frame_opts = fopts
hires_mfcc_opts.num_ceps = 40
hires_mfcc_opts.mel_opts = hires_mb_opts
hires_mfcc_opts.use_energy = False


def lid_module(key, audio_file, start, end):
    # ==================================
    #       Get data and process it.
    # ==================================
    wav_spc = "scp:echo " + key + " 'sox -V0 -t wav " + audio_file + " -c 1 -r 16000 -t wav - trim " + str(
        start) + " " + str(
        float(end) - float(start)) + "|' |"
    hires_mfcc = Mfcc(hires_mfcc_opts)
    wav = SequentialWaveReader(wav_spc).value()
    hi_feat = hires_mfcc.compute_features(wav.data()[0], wav.samp_freq, 1.0)
    hi_feat = hi_feat.numpy() - CMVN
    X = hi_feat.T
    print(X.shape)
    if X.shape[1] >= 384:
        X = np.expand_dims(X[:,:384], 0)
    else:
        padded_x = torch.zeros(40, 384)
        padded_x[:,:X.shape[1]]	 = torch.from_numpy(X)
        X = np.expand_dims(padded_x, 0)
    print(X.shape)
    emb = nn_LID_model_DA.emb(torch.from_numpy(X))[0]
    print(emb.shape)
    #np.save("emb/"+key+".npy",emb.numpy())
#    smax = torch.nn.functional.softmax(v)
#    print(smax)
#    print(i2l[torch.argmax(smax).item()])
#    print(v)
    # # print(v.shape)
    # v = Vector(v[0, :])
    # v.add_vec_(-1.0, mean)
    # rows, cols = lda.num_rows, lda.num_cols
    # vec_dim = v.dim
    # vec_out = Vector(150)
    # vec_out.copy_col_from_mat_(lda, vec_dim)
    # vec_out.add_mat_vec_(1.0, lda.range(0, rows, 0, vec_dim), MatrixTransposeType.NO_TRANS, v, 1.0)
    # norm = vec_out.norm(2.0)
    # ratio = norm / math.sqrt(vec_out.dim)
    # vec_norm = vec_out.scale_(1.0 / ratio)
    # output = language_classifier.get_log_posteriors_vector(vec_norm)
    # print(i2l[output.max_index()[1]])


if __name__ == "__main__":
    # wavs = glob.glob(args.data_path + '*/*.wav')
    wavs = ["telugu.wav"]
    for audio in wavs:
        utt = audio.split('/')[-1][:-4]
        Segments, _ = save_segs.build_response(audio)
        for segment in Segments:
            seg_key = utt + "-" + str("{0:.2f}".format(float(segment[0]) / 100)).zfill(7).replace(".", "") + "-" + str(
                "{0:.2f}".format(float(segment[1]) / 100)).zfill(7).replace(".", "")
            start_time = float(segment[0]) / 100
            end_time = float(segment[1]) / 100
            if end_time - start_time >= 3:
                data = {"AudioFile": audio, "startTime": start_time, "endTime": end_time}
                lid_module(seg_key, data["AudioFile"], data["startTime"], data["endTime"])
