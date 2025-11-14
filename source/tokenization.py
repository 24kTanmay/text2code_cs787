#!/bin/bash python

# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import copy
import logging

logger = logging.getLogger(__name__)


def tokenization_function(examples, tokenizer=None, max_seq_length=1024, new_id_map=None):
	"""
	Tokenization function: Preparing the input
	"""
	my_examples = {"input_ids": [], "code_mask": [], "docstr_mask": [], "special_tokens_mask": [],
				   "prefix_lm_token_idx": [], "length": []}

	for _id, docstring, code in zip(examples["_id"], examples["docstring"], examples["code"]):
		pre_docstring_ids = [tokenizer.convert_tokens_to_ids('<comments>')]
		docstring_ids = tokenizer.encode(docstring, add_special_tokens=False)
		code_ids = tokenizer.encode(code, add_special_tokens=False)

		if new_id_map is not None:
			new_code_ids = []
			for cid in code_ids:
				if cid in new_id_map:
					new_code_ids.append(new_id_map[cid])
				else:
					new_code_ids.append(cid)
			code_ids = copy.deepcopy(new_code_ids)

		pre_code_ids = [tokenizer.convert_tokens_to_ids('<python>')]
		end_of_seq = [tokenizer.convert_tokens_to_ids('<eot>')]

		token_ids = pre_docstring_ids + docstring_ids + pre_code_ids + code_ids + end_of_seq

		# Masks
		code_mask = [0] * len(pre_docstring_ids + docstring_ids + pre_code_ids) + \
					[1] * len(code_ids + end_of_seq)
		docstring_mask = [1] * len(pre_docstring_ids + docstring_ids + pre_code_ids) + \
						 [0] * len(code_ids + end_of_seq)
		special_tokens_mask = [1] * len(pre_docstring_ids) + \
							  [0] * len(docstring_ids) + \
							  [1] * len(pre_code_ids) + \
							  [0] * len(code_ids) + \
							  [1] * len(end_of_seq)

		assert len(code_mask) == len(token_ids) == len(docstring_mask) == len(special_tokens_mask)

		if len(token_ids) <= max_seq_length:
			# Filter too long inputs!
			my_examples['input_ids'].append(token_ids)
			my_examples['code_mask'].append(code_mask)
			my_examples['docstr_mask'].append(docstring_mask)
			my_examples['special_tokens_mask'].append(special_tokens_mask)
			my_examples['length'].append(len(token_ids))

			# If we want a prefix lm, provide the prefix_lm_token_id accordingly
			# Basically, give the token up-to-which you want to consider as prefix
			my_examples['prefix_lm_token_idx'].append(len(pre_docstring_ids + docstring_ids))

		else:
			logger.info(f'Rejecting example {_id} with length {len(token_ids)} > {max_seq_length}')

	return my_examples


def tokenization_function_raw(examples, tokenizer=None, max_seq_length=1024, new_id_map=None):
	"""
	Tokenization function: Preparing the input
	"""
	my_examples = {"input_ids": [], "code_mask": [], "docstr_mask": [], "special_tokens_mask": [],
				   "prefix_lm_token_idx": [], "length": []}

	for _id, docstring, code in zip(examples["_id"], examples["docstring"], examples["code"]):
		# Handle empty docstring
		pre_docstring_ids = [tokenizer.convert_tokens_to_ids('<comments>')]
		if "\n" in docstring:
			doc = docstring.split('\n')
			new_doc = []
			for d in doc:
				new_doc.append(f"    {d}")
			new_docstring = '\n'.join(new_doc)
			docstring_ids = tokenizer.encode('\n    \"\"\"\n' + new_docstring + '\n    \"\"\"', add_special_tokens=False)
		else:
			docstring_ids = tokenizer.encode('\n    \"\"\" ' + docstring + ' \"\"\"', add_special_tokens=False)

		# Separate signature from the data:
		end_of_sign = code.find(': <NEW_LINE> <INDENT>')
		
		if end_of_sign == -1:
			# Marker not found, treat as if there is no signature
			#logger.info(f"Signature marker not found for example {_id}. Processing with empty signature.")
			signature = ""
			# 'code' variable already contains the full code, so no change needed
		else:
			# Marker was found, try to extract signature
			signature_candidate = code[:end_of_sign + 1]
			
			if 'def ' in signature_candidate:
				signature = signature_candidate[signature_candidate.find('def '):]
				code = code[end_of_sign + 1:] # Use code *after* the signature
			elif 'class ' in signature_candidate:
				signature = signature_candidate[signature_candidate.find('class '):]
				code = code[end_of_sign + 1:] # Use code *after* the signature
			else:
				# Marker found, but no 'def' or 'class'. Treat as empty signature.
				#logger.warning(f"Signature marker found but no 'def'/'class' for example {_id}. Processing with empty signature.")
				signature = ""
				# 'code' variable already contains the full code, so no change needed


		code_ids = tokenizer.encode(code, add_special_tokens=False)
		pre_sign_ids = [tokenizer.convert_tokens_to_ids('<|beginoftext|>')]
		signature_ids = tokenizer.encode(signature, add_special_tokens=False)

		if new_id_map is not None:
			new_code_ids = []
			for cid in code_ids:
				if cid in new_id_map:
					new_code_ids.append(new_id_map[cid])
				else:
					new_code_ids.append(cid)
			code_ids = copy.deepcopy(new_code_ids)

			new_sign_ids = []
			for sid in signature_ids:
				if sid in new_id_map:
					new_sign_ids.append(new_id_map[sid])
				else:
					new_sign_ids.append(sid)
			signature_ids = copy.deepcopy(new_sign_ids)

		pre_code_ids = [tokenizer.convert_tokens_to_ids('<python>')]
		end_of_seq = [tokenizer.convert_tokens_to_ids('<eot>')]

		token_ids = pre_sign_ids + signature_ids + pre_docstring_ids + docstring_ids + pre_code_ids + code_ids + end_of_seq

		# Masks
		code_mask = [1] * len(pre_sign_ids + signature_ids) + \
					[0] * len(pre_docstring_ids + docstring_ids + pre_code_ids) + \
					[1] * len(code_ids + end_of_seq)
		docstring_mask = [1] * len(pre_sign_ids + signature_ids + pre_docstring_ids + docstring_ids + pre_code_ids) + \
						 [0] * len(code_ids + end_of_seq)
		special_tokens_mask = [1] * len(pre_sign_ids) + \
							  [0] * len(signature_ids) + \
							  [1] * len(pre_docstring_ids) + \
							  [0] * len(docstring_ids) + \
							  [1] * len(pre_code_ids) + \
							  [0] * len(code_ids) + \
							  [1] * len(end_of_seq)

		assert len(code_mask) == len(token_ids) == len(docstring_mask) == len(special_tokens_mask)

		# Truncate if sequence is too long
		if len(token_ids) > max_seq_length:
			#logger.info(f'Truncating example {_id} with length {len(token_ids)} to {max_seq_length}')
			token_ids = token_ids[:max_seq_length]
			code_mask = code_mask[:max_seq_length]
			docstring_mask = docstring_mask[:max_seq_length]
			special_tokens_mask = special_tokens_mask[:max_seq_length]

		# Always append the (potentially truncated) example
		my_examples['input_ids'].append(token_ids)
		my_examples['code_mask'].append(code_mask)
		my_examples['docstr_mask'].append(docstring_mask)
		my_examples['special_tokens_mask'].append(special_tokens_mask)
		my_examples['length'].append(len(token_ids))

		# If we want a prefix lm, provide the prefix_lm_token_id accordingly
		# Cap the prefix index at max_seq_length in case of truncation
		prefix_idx = len(pre_sign_ids + signature_ids + pre_docstring_ids + docstring_ids)
		my_examples['prefix_lm_token_idx'].append(min(prefix_idx, max_seq_length))

	return my_examples
