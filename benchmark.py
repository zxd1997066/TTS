# Based on Jupyter notebook from:
# https://github.com/mozilla/TTS/blob/20a6ab3d612eea849a49801d365fcd071839b7a1/notebooks/Benchmark.ipynb

import os
import sys
import io
import torch
import time
import json
import numpy as np
from collections import OrderedDict
from matplotlib import pylab as plt

import librosa
import librosa.display

from TTS.models.tacotron import Tacotron
from TTS.layers import *
from TTS.utils.data import *
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text import text_to_sequence
from TTS.utils.synthesis import synthesis, text_to_seqvec, id_to_torch
from TTS.utils.visual import visualize
from TTS.utils.text.symbols import symbols, phonemes

import argparse
import csv

def main():
    args = parse_args()
    if args.precision == "bfloat16":
        # with torch.amp.autocast(enabled=True, configure=torch.bfloat16, torch.no_grad(): 
        print("Running with bfloat16...")

    # Set constants
    model_path = args.model_path
    config_path = args.config_path
    config = load_config(config_path)
    config['batch_size'] = args.batch_size
    config['eval_batch_size'] = args.batch_size
    print(config)
    dataset_name = args.dataset_name
    metadata_path = args.metadata_path
    test_sentences = load_test_sentences(dataset_name, metadata_path)
    use_cuda = False
    # Set some config fields manually for testing
    config.use_forward_attn = True
    # Use only one speaker
    speaker_id = None
    speakers = []
    # load the audio processor
    ap = AudioProcessor(**config.audio)
    # Load TTS model
    model = load_model(config, speakers, model_path)
    # Run inference
    if args.channels_last:
        model_oob = model
        model.to(memory_format=torch.channels_last)
        model = model_oob
        print("---- Use channels last format.")

    if args.ipex:
        import intel_extension_for_pytorch as ipex
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
            print('---- Run IPEX model with bfloat16...')
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
            print('---- Run IPEX model with float32...')

    if args.jit:
        jit_inputs = [torch.randint(args.batch_size, 128, (1, 122)),
                      torch.Tensor([122]),
                      torch.rand(args.batch_size, 1, 80)]
        model = torch.jit.trace(model, jit_inputs, check_trace=False)
        print("---- With JIT enabled.")
        if args.ipex:
            model = torch.jit.freeze(model)
    run_inference(args, test_sentences, model, config, use_cuda, ap, speaker_id)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model-path", help="Tacotron model filename", action="store", required=True)
    parser.add_argument("--config-path", help="Tacotron config filename", action="store", required=True)
    parser.add_argument("--dataset-name", help="Dataset name", action="store", default="dummy", required=False)
    parser.add_argument("--perf-num-iters", type=int, default=0)
    parser.add_argument("--perf-num-warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metadata-path", help="Dataset metadata filename (file containing sentences)",
                        default="./dummy_data.csv", action="store", required=False)
    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    parser.add_argument('--precision', default="float32",
                        help='precision, "float32" or "bfloat16"')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='convert model to script model')
    parser.add_argument('--channels_last', type=int, default=1,
                        help='use channels last format')
    parser.add_argument('--profile', action='store_true',
                        help='Trigger profile on current topology.')
    args = parser.parse_args()
    return args

def load_test_sentences(dataset_name, metadata_path):
    dataset_load_function = {
        "dummy": get_dummy_data,
        "ljspeech": get_ljspeech_data
    }
    with open(metadata_path, encoding='utf-8') as filestream:
        raw_data = filestream.readlines()
    return dataset_load_function.get(dataset_name, lambda x: [])(raw_data)

def get_dummy_data(raw_data):
    reader = csv.reader(raw_data)
    data = [element for row in reader for element in row if element]
    return data

def get_ljspeech_data(raw_data):
    reader = csv.reader(raw_data, delimiter="|", quoting=csv.QUOTE_NONE)
    data = [row[2] for row in reader]
    return data

def load_model(config, speakers, model_path):
    # Only one speaker
    speakers = []
    # load the model
    num_chars = len(phonemes) if config.use_phonemes else len(symbols)
    model = setup_model(num_chars, len(speakers), config)
    # load model state
    cp = torch.load(model_path, map_location=lambda storage, loc: storage)
    # load the model
    model.load_state_dict(cp['model'])
    model.eval()
    # set model stepsize
    if 'r' in cp:
        model.decoder.set_r(cp['r'])
    model.decoder.max_decoder_steps = 2000
    return model

def run_inference(args, test_sentences, model, config, use_cuda, ap, speaker_id):
    num_sentences = len(test_sentences)
    throughputs = []
    latencies = []
    avg_throughput = 0
    avg_latency = 0
    batch_time_list = []
    with torch.no_grad():
        if args.precision == "bfloat16":
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                for i, sentence in enumerate(test_sentences):
                    if args.perf_num_iters != 0 and i >= args.perf_num_iters:
                        break
                    # write_progress_bar(i, num_sentences, avg_throughput, avg_latency)
                    throughput_s, latency_ms, prof = tts(args, model, sentence, config, use_cuda, ap, speaker_id, figures=True)
                    print("Iteration: {}, inference time: {} sec.".format(i, latency_ms/1000), flush=True)
                    if i < args.perf_num_warmup:
                        continue
                    batch_time_list.append(latency_ms)
                    throughputs.append(throughput_s)
                    latencies.append(latency_ms)
                    if i == int(args.perf_num_iters/2) and args.profile:
                        import pathlib
                        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                        if not os.path.exists(timeline_dir):
                            os.makedirs(timeline_dir)
                        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                    "tts" + str(i) + '-' + str(os.getpid()) + '.json'
                        print(timeline_file)
                        prof.export_chrome_trace(timeline_file)
                        table_res = prof.key_averages().table(sort_by="cpu_time_total")
                        print(table_res)
                        # save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
        else:
            for i, sentence in enumerate(test_sentences):
                if args.perf_num_iters != 0 and i >= args.perf_num_iters:
                    break
                # write_progress_bar(i, num_sentences, avg_throughput, avg_latency)
                throughput_s, latency_ms, prof = tts(args, model, sentence, config, use_cuda, ap, speaker_id, figures=True)
                print("Iteration: {}, inference time: {} sec.".format(i, latency_ms/1000), flush=True)
                if i < args.perf_num_warmup:
                    continue
                batch_time_list.append(latency_ms)
                throughputs.append(throughput_s)
                latencies.append(latency_ms)
                if i == int(args.perf_num_iters/2) and args.profile:
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        os.makedirs(timeline_dir)
                    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                "tts" + str(i) + '-' + str(os.getpid()) + '.json'
                    print(timeline_file)
                    prof.export_chrome_trace(timeline_file)
                    table_res = prof.key_averages().table(sort_by="cpu_time_total")
                    print(table_res)
                    # save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
        avg_throughput = sum(throughputs)/len(throughputs)
        avg_latency = sum(latencies)/len(latencies)
        print(avg_latency)
        # close_progress_bar()
        print("\n", "-"*20, "Summary", "-"*20)
        print("inference latency:\t {:.3f} ms".format(avg_latency))
        print("inference Throughput:\t {:.2f} samples/s".format(avg_throughput))
        # P50
        batch_time_list.sort()
        p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
        p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
        p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
        print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
                % (p50_latency, p90_latency, p99_latency))

        return

def write_progress_bar(i, iterations, avg_throughput, avg_latency, toolbar_width=50):
    num_bars = round(toolbar_width*i/iterations)
    line = ["[", u"\u2588"*num_bars, " "*(toolbar_width-num_bars), "]",
            " ", str(i), "/", str(iterations),
            " ", "|", " ", "avg thrpt: ", f"{avg_throughput:.2f}", " [elem/s]",
            " ", "|", " ", "avg lat: ", f"{avg_latency:.2f}", " [ms/elem]"]
    line_str = "".join(line)
    line_width = len(line_str)
    sys.stdout.write(str(line_str.encode('utf-8')))
    sys.stdout.flush()
    sys.stdout.write("\b" * line_width)
    sys.stdout.flush()

def close_progress_bar():
    sys.stdout.write("\n")
    sys.stdout.flush()

def tts(args, model, text, CONFIG, use_cuda, ap, speaker_id, figures=True):
    t_1 = time.time()
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
            waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs_size = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, False, CONFIG.enable_eos_bos_chars)
    else:
        prof = ""
        waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs_size = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, False, CONFIG.enable_eos_bos_chars)
    mel_postnet_spec = ap._denormalize(mel_postnet_spec)
    mel_postnet_spec = ap._normalize(mel_postnet_spec)
    run_time = time.time() - t_1
    latency_s = run_time / inputs_size[1]
    latency_ms = latency_s * 1000
    throughput_s = 1 / latency_s
    return throughput_s, latency_ms, prof

def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()

if __name__ == "__main__":
    sys.exit(main())
