import json
import random
import string
import time

import numpy as np
import tritonclient.grpc as client_util
from tokenizers import Tokenizer
from tritonclient.utils import np_to_triton_dtype, InferenceServerException

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))


class CodeGenProxy:
    def __init__(self, host: str = 'triton', port: int = 8001, verbose: bool = False):
        self.tokenizer = Tokenizer.from_file('/python-docker/cgtok/tokenizer.json')
        self.client = client_util.InferenceServerClient(url=f'{host}:{port}', verbose=verbose)
        self.PAD_CHAR = 50256

        # Max number of tokens the model can handle
        self.MAX_MODEL_LEN = 2048

    class TokensExceedsMaximum(Exception):
        pass

    @staticmethod
    def prepare_tensor(name: str, tensor_input):
        t = client_util.InferInput(
            name, tensor_input.shape, np_to_triton_dtype(tensor_input.dtype))
        t.set_data_from_numpy(tensor_input)
        return t

    @staticmethod
    def trim_with_stopwords(output: str, stopwords: list) -> str:
        for w in sorted(stopwords, key=len, reverse=True):
            if output.endswith(w):
                output = output[:-len(w)]
                break
        return output

    @staticmethod
    def to_word_list_format(word_dict, tokenizer):
        flat_ids = []
        offsets = []
        for word_dict_item in word_dict:
            item_flat_ids = []
            item_offsets = []

            for word in word_dict_item:
                ids = tokenizer.encode(word).ids

                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

                # Hack, can we do this better?
                if word == '\n\n':
                    item_flat_ids += [198, 198]
                    item_offsets.append(2)

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

    def generate(self, data):
        model_name = "fastertransformer"
        prompt = data['prompt']
        n = data.get('n', 1)
        input_start_ids = np.expand_dims(self.tokenizer.encode(prompt).ids, 0)
        input_start_ids = np.repeat(input_start_ids, n, axis=0).astype(np.uint32)
        prompt_len = input_start_ids.shape[1]
        input_len = prompt_len * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        max_tokens = data.get('max_tokens', 16)
        prompt_tokens: int = input_len[0][0]
        requested_tokens = max_tokens + prompt_tokens
        if requested_tokens > self.MAX_MODEL_LEN:
            print(1)
            raise self.TokensExceedsMaximum(
                f"This model's maximum context length is {self.MAX_MODEL_LEN}, however you requested "
                f"{requested_tokens} tokens ({prompt_tokens} in your prompt; {max_tokens} for the completion). "
                f"Please reduce your prompt; or completion length."
            )
        output_len = np.ones_like(input_len).astype(np.uint32) * max_tokens
        num_logprobs = data.get('logprobs', -1)
        if num_logprobs is None:
            num_logprobs = 1
        want_logprobs = num_logprobs > 0

        temperature = data.get('temperature', 0.2)
        if temperature == 0.0:
            temperature = 1.0
            top_k = 1
        else:
            top_k = data.get('top_k', 0)

        top_p = data.get('top_p', 1.0)
        frequency_penalty = data.get('frequency_penalty', 1.0)
        runtime_top_k = top_k * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        runtime_top_p = top_p * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        random_seed = np.random.randint(0, 2 ** 31 - 1, (input_start_ids.shape[0], 1), dtype=np.int32)
        temperature = temperature * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        repetition_penalty = frequency_penalty * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        is_return_log_probs = want_logprobs * np.ones([input_start_ids.shape[0], 1]).astype(np.bool_)
        beam_width = (1 * np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
        start_ids = self.PAD_CHAR * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        end_ids = self.PAD_CHAR * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)

        stop_words = data.get('stop', [])
        if stop_words is None:
            stop_words = []
        if stop_words:
            stop_word_list = np.repeat(self.to_word_list_format([stop_words], self.tokenizer), input_start_ids.shape[0],
                                       axis=0)
        else:
            stop_word_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
                np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)

        # Not used
        bad_words_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
            np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)

        # Get beam parameters from input
        beam_width[0][0] = data['beam_width']
        beam_search_diversity_rate[0][0] = data['beam_search_diversity_rate']
        # print(f"\n\nbeam width is {beam_width[0][0]}")
        # print(f"beam search diversity rate is {beam_search_diversity_rate[0][0]}\n\n")
        inputs = [
            self.prepare_tensor("input_ids", input_start_ids),
            self.prepare_tensor("input_lengths", input_len),
            self.prepare_tensor("request_output_len", output_len),
            self.prepare_tensor("runtime_top_k", runtime_top_k),
            self.prepare_tensor("runtime_top_p", runtime_top_p),
            self.prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
            self.prepare_tensor("random_seed", random_seed),
            self.prepare_tensor("temperature", temperature),
            self.prepare_tensor("len_penalty", len_penalty),
            self.prepare_tensor("repetition_penalty", repetition_penalty),
            self.prepare_tensor("is_return_log_probs", is_return_log_probs),
            self.prepare_tensor("beam_width", beam_width),
            self.prepare_tensor("start_id", start_ids),
            self.prepare_tensor("end_id", end_ids),
            self.prepare_tensor("bad_words_list", bad_words_list),
            self.prepare_tensor("stop_words_list", stop_word_list),
        ]

        result = self.client.infer(model_name, inputs)

        output_data = result.as_numpy("output_ids")
        if output_data is None:
            raise RuntimeError("No output data")


        # Calculate the beam index with the highest log prob in constant time.
        one_beam = False
        print(f"one_beam is: {one_beam}")
        lp_data = result.as_numpy("output_log_probs")
        lp_sums = np.zeros((lp_data.shape[0], lp_data.shape[1]))
        lp_result = np.zeros((lp_data.shape[0], lp_data.shape[2]))  # Pick the best one from each beam
        data_result = np.zeros((output_data.shape[0], output_data.shape[2]), dtype=int)
        # sequence_lengths has shape [n, beam_width]
        sequence_lengths = result.as_numpy("sequence_length")
        # for i in range(lp_data.shape[0]):  # iterate over batch
        #     for j in range(lp_data.shape[1]):  # iterate over beam
        #         data = lp_data[i, j, :]
        #         lp_sums[i, j] = np.sum(data) / np.count_nonzero(data)
        #     best_beam = np.argmax(lp_sums[i, :])
        #     lp_result[i, :] = lp_data[i, best_beam, :]
        #     data_result[i, :] = output_data[i, best_beam, :]
        #     sequence_lengths = sequence_lengths[:, best_beam]

        # Assume the best beam is 0. (Randomly choose a beam)
        # print("\n\n\nChoosing random beam\n\n\n")
        if one_beam:
            lp_result[:, :] = lp_data[:, 0, :]
            lp_data = lp_result
            data_result[:, :] = output_data[:, 0, :]
            output_data = data_result
            sequence_lengths = sequence_lengths[:, 0]
            # output_data = output_data.squeeze(1)

        if want_logprobs:
            print("want logprobs")
            # clp_data = result.as_numpy("cum_log_probs").squeeze(1)
        else:
            lp_data = [None] * output_data.shape[0]

        # input_len is the same across beams so this squeeze is fine.
        gen_len = sequence_lengths - input_len.squeeze(1)
        print(f"output_data.shape is: {output_data.shape}")
        print(f"gen_len.shape is: {gen_len.shape}")
        print(f"lp_data.shape is: {lp_data.shape}")
        if one_beam:
            # output_data.shape is: (1, max_len+9)
            # gen_len.shape is: (1,)
            # lp_data.shape is: (1, max_len)
            decoded = self.tokenizer.decode_batch(
                [out[prompt_len:prompt_len + g] for g, out in zip(gen_len, output_data)])
            trimmed = [self.trim_with_stopwords(d, stop_words) for d in decoded]
        else:
            # output_data.shape is: (1, beam_width, max_len+9)
            # gen_len.shape is: (1, beam_width)
            # lp_data.shape is: (1, beam_width, max_len)
            decoded = []; trimmed = []
            bw = output_data.shape[1]
            for i in range(bw):
                decoded.append(self.tokenizer.decode_batch(
                    [out[prompt_len:prompt_len + g] for g, out in zip(gen_len[:,i], output_data[:,i,:])]))
                trimmed.append([self.trim_with_stopwords(d, stop_words) for d in decoded[i]])
        print(f"decoded type is: {type(decoded)}; shape is: {len(decoded)}, 0-th value is {decoded[0]}")
        print(f"trimmed has shape: {len(trimmed)}; is: {trimmed}")

        if not one_beam:
            output_data = output_data.squeeze(0)
            lp_data = lp_data.squeeze(0)
            gen_len = gen_len.squeeze(0)
        choices = []
        for i, (text, tokens, lps, g) in enumerate(zip(trimmed, output_data, lp_data, gen_len)):
            print(f"tokens shape is {tokens.shape}")
            print(f"lps shape is {lps.shape}")
            reason = "length" if max_tokens == g else "stop"
            if lps is not None:
                tokens_str = [self.tokenizer.decode([t]) for t in tokens[prompt_len:prompt_len + g]]
                offsets = [len(prompt)] + (np.cumsum([len(t) for t in tokens_str]) + len(prompt)).tolist()[:-1]

                # Fake some log probs for top_logprobs
                top_logprobs = []
                for ii, t in enumerate(tokens_str):
                    fakedict = {}
                    top_token_lp = float(lps[ii])
                    fakedict[t] = top_token_lp
                    while len(fakedict) < num_logprobs:
                        random_token = random.randint(0, self.tokenizer.get_vocab_size() - 1)
                        random_token_str = self.tokenizer.decode([random_token])
                        if random_token_str in fakedict:
                            continue
                        random_token_lp = top_token_lp - random.random()
                        fakedict[random_token_str] = random_token_lp
                    top_logprobs.append(fakedict)

                lpdict = {
                    'token_logprobs': lps.tolist(),
                    'top_logprobs': top_logprobs,
                    'tokens': tokens_str,
                    'text_offset': offsets,
                }
            else:
                lpdict = None

            choice = {
                'text': text,
                'index': i,
                'finish_reason': reason,
                'logprobs': lpdict,
            }
            choices.append(choice)

        completion = {
            'id': None,  # fill in
            'model': 'codegen',
            'object': 'text_completion',
            'created': int(time.time()),
            'choices': None,  # fill in
            'usage': {
                'completion_tokens': int(gen_len.sum()),
                'prompt_tokens': int(prompt_len),
                'total_tokens': int(gen_len.sum() + prompt_len),
            }
        }
        return completion, choices, lp_data

    @staticmethod
    def random_completion_id():
        return 'cmpl-' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29))

    def streamed_response(self, completion, choices):
        for c in choices:
            completion['id'] = self.random_completion_id()
            completion['choices'] = [c]
            yield f'data: {json.dumps(completion)}\n\n'
        yield 'data: [DONE]\n\n'

    def non_streamed_response(self, completion, choices) -> str:
        completion['id'] = self.random_completion_id()
        completion['choices'] = choices
        return json.dumps(completion)

    def __call__(self, data: dict):
        st = time.time()
        try:
            completion, choices = self.generate(data)
        except InferenceServerException as E:
            print(E)
            completion = {}
            choices = []
        ed = time.time()
        print(f"Returned completion in {(ed - st) * 1000} ms")
        if data.get('stream', False):
            return self.streamed_response(completion, choices)
        else:
            return self.non_streamed_response(completion, choices)
