# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """
from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')

import logging
import os
from io import open
import tensorflow as tf
import json
import sys
import collections
import six
import tokenization
logger = logging.getLogger(__name__)




class DocvqaExample(object):
    """A single training/test example for token classification."""


    def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False,
               boxes = []):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.boxes = boxes


class InputFeatures(object):
    """A single set of features of data."""

   
    def __init__(self,
               unique_id,
               qas_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_positions=None,
               end_positions=None,
               is_impossible=None,
               boxes = None,
               p_mask =None):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.is_impossible = is_impossible
        self.boxes = boxes
        self.p_mask = p_mask
def read_docvqa_examples(input_file, is_training, skip_match_answers=True):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False
  count_match = 0
  count_nomatch = 0

  examples = []
  for paragraph in input_data:
      image_id =  paragraph["image_id"]
      paragraph_text = paragraph["context"]
      boxes = paragraph["boxes"]
      doc_tokens = paragraph["context"]
      for qa in paragraph["qas"]:
        qas_id = qa["qid"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        answer = qa["answer"][0]        
        orig_answer_text = answer["text"]
        if is_training:
          if not is_impossible:
            answer = qa["answer"][0]
            orig_answer_text = answer["text"]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            start_position = qa["answer"][0]["answer_start"]
            end_position = qa["answer"][0]["answer_end"]
            actual_text = " ".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if not skip_match_answers:
                if actual_text.find(cleaned_answer_text) == -1:
                  tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                 actual_text, cleaned_answer_text)
                  count_nomatch+=1
                  continue
            count_match+=1
          else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""

        example =DocvqaExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible,
            boxes=boxes)
        examples.append(example)
  return examples


def convert_examples_to_features(examples,label_list, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training, 
                                 pad_token_label_id=-100):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  features = []
  label_map = {label: i for i, label in enumerate(label_list)}
  query_label_ids = []
  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)
    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]
    query_label_ids=[0]+[pad_token_label_id] * (len(query_tokens) - 1)
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    all_doc_boxes_tokens = []
    cls_token_box=[0, 0, 0, 0]
    sep_token_box=[1000, 1000, 1000, 1000]
    pad_token_box=[0, 0, 0, 0] 
    ques_token_box=[0, 0, 0, 0]
    all_label_ids = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      box = example.boxes[i]
      if i == example.start_position:
            lab = 1
      elif i == example.end_position:
            lab = 2
      else:   
            lab = 0
      p = [lab] + [pad_token_label_id] * (len(sub_tokens) - 1)
      all_label_ids+=p
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)
        all_doc_boxes_tokens.append(box)
        #p = [lab] + [pad_token_label_id] * (len(sub_tokens) - 1)
        #all_label_ids+=p        
            


    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if is_training and not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)
    
    #TODO Remove later
    #if len(doc_spans)>1:
    #    continue

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      boxes_tokens = []
      label_ids = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      p_mask = []
      tokens.append("[CLS]")
      p_mask.append(0)
      boxes_tokens.append(cls_token_box)
      segment_ids.append(0)
      label_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        boxes_tokens.append(ques_token_box)
        segment_ids.append(0)
        p_mask.append(1)
      label_ids=label_ids+query_label_ids
      tokens.append("[SEP]")
      p_mask.append(1)
      boxes_tokens.append(sep_token_box)
      segment_ids.append(0)
      label_ids.append(0)
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        label_ids.append(all_label_ids[split_token_index])
        boxes_tokens.append(all_doc_boxes_tokens[split_token_index]) 
        segment_ids.append(1)
        p_mask.append(0)
      tokens.append("[SEP]")
      p_mask.append(1)
      boxes_tokens.append(sep_token_box)
      segment_ids.append(1)
      label_ids.append(pad_token_label_id)
      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)
      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        boxes_tokens.append(pad_token_box)
        label_ids.append(pad_token_label_id)
        p_mask.append(1)
      
      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(boxes_tokens) == max_seq_length
      assert len(label_ids) == max_seq_length      
      assert len(p_mask) == max_seq_length
     
      start_position = None
      end_position = None
      if is_training and not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and example.is_impossible:
        start_position = 0
        end_position = 0
      #label_ids = [-1]*max_seq_length
      #if is_training and (start_position!=0 and end_position!=0):
      #  label_ids[start_position]=0
      #  label_ids[end_position]=1

      if example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % (unique_id))
        tf.logging.info("example_index: %s" % (example_index))
        tf.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and example.is_impossible:
          tf.logging.info("impossible example")
        if is_training and not example.is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d" % (start_position))
          tf.logging.info("end_position: %d" % (end_position))
          tf.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))
      feature = InputFeatures(
          unique_id=unique_id,
          qas_id=example.qas_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_positions=start_position,
          end_positions=end_position,
          is_impossible=example.is_impossible,
          boxes=boxes_tokens,
          p_mask = p_mask,
          )
      features.append(feature)
      '''
      print(feature)
      print('unique_id',feature.unique_id)
      print('tokens',feature.tokens)
      print('example_index',feature.example_index)
      print('input_ids',feature.input_ids)
      print('segment_ids',feature.segment_ids)
      print('doc_span_index',feature.doc_span_index)
      print('token_to_orig_map',feature.token_to_orig_map)
      print('token_is_max_contex',feature.token_is_max_context)
      print('input_mask',feature.input_mask)
      print('start_position',feature.start_position)
      print('end_position',feature.end_position)
      print('is_impossible',feature.is_impossible)'''
      # Run callback
      #output_fn(feature)

      unique_id += 1
  return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index
