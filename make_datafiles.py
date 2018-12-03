import sys
import os
import hashlib
import subprocess
import collections
import unicodedata
import json
import tarfile
import io
import pickle as pkl
import tensorflow as tf
import re

finished_files_dir = "../finished_files"
num_tokens_per_sentence = 50
num_sentence = 20

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
  
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z]+", " ", w)
    
    w = w.rstrip().strip()
    
    return w
    
if __name__ == '__main__':

    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # generate eval tar
    txt_file = 'gs://bytecup2018/bytecup2018/bytecup.corpus.eval.txt'
    out_file =  os.path.join(finished_files_dir, "val.tar")
    with tf.gfile.Open(txt_file, "r") as f:
        with tarfile.open(out_file, 'w') as writer:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                print("Writing story {} of {}; {:.2f} percent done".format(idx, len(lines), float(idx)*100.0/float(len(lines))))
                id_num = json.loads(line)['id']
                article = preprocess_sentence(json.loads(line)['content']).split()
                abstract_sents = [preprocess_sentence(json.loads(line)['title'])]
                #print(article)
                article_sents = [' '.join(article[i:i+num_tokens_per_sentence]) for i in range(0, len(article), num_tokens_per_sentence)]
                # Write to JSON file
                js_example = {}
                js_example['id'] = id_num
                js_example['article'] = article_sents
                js_example['abstract'] = abstract_sents
                js_serialized = json.dumps(js_example, indent=4).encode()
                save_file = io.BytesIO(js_serialized)
                tar_info = tarfile.TarInfo('{}/{}.json'.format(
                    os.path.basename(out_file).replace('.tar', ''), idx))
                tar_info.size = len(js_serialized)
                writer.addfile(tar_info, save_file)

    # generate test tar
    txt_file = 'gs://bytecup2018/bytecup2018/bytecup.corpus.test.txt'
    out_file =  os.path.join(finished_files_dir, "test.tar")
    with tf.gfile.Open(txt_file, "r") as f:
        with tarfile.open(out_file, 'w') as writer:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                print("Writing story {} of {}; {:.2f} percent done".format(idx, len(lines), float(idx)*100.0/float(len(lines))))
                id_num = json.loads(line)['id']
                article = preprocess_sentence(json.loads(line)['content']).split()
                abstract_sents = [preprocess_sentence(json.loads(line)['title'])]
                #print(article)
                article_sents = [' '.join(article[i:i+num_tokens_per_sentence]) for i in range(0, len(article), num_tokens_per_sentence)]
                # Write to JSON file
                js_example = {}
                js_example['id'] = id_num
                js_example['article'] = article_sents
                js_example['abstract'] = abstract_sents
                js_serialized = json.dumps(js_example, indent=4).encode()
                save_file = io.BytesIO(js_serialized)
                tar_info = tarfile.TarInfo('{}/{}.json'.format(
                    os.path.basename(out_file).replace('.tar', ''), idx))
                tar_info.size = len(js_serialized)
                writer.addfile(tar_info, save_file)
    
    # generate train tar and vocab file
    train_files_size = 2
    out_file =  os.path.join(finished_files_dir, "train.tar")
    vocab_counter = collections.Counter()
    with tarfile.open(out_file, 'w') as writer:
        for i in range(train_files_size):
            txt_file = 'gs://bytecup2018/bytecup2018/bytecup.corpus.train.{}.txt'.format(i)
            with tf.gfile.Open(txt_file, "r") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    print("Writing story {} of {}; {:.2f} percent done".format(idx, len(lines), float(idx)*100.0/float(len(lines))))
                    id_num = json.loads(line)['id']
                    article = preprocess_sentence(json.loads(line)['content']).split()
                    abstract_sents = [preprocess_sentence(json.loads(line)['title'])]
                    #print(article)
                    article_sents = [' '.join(article[i:i+num_tokens_per_sentence]) for i in range(0, len(article), num_tokens_per_sentence)]
                    # Write to JSON file
                    js_example = {}
                    js_example['id'] = id_num
                    js_example['article'] = article_sents
                    js_example['abstract'] = abstract_sents
                    js_serialized = json.dumps(js_example, indent=4).encode()
                    save_file = io.BytesIO(js_serialized)
                    tar_info = tarfile.TarInfo('{}/{}.json'.format(
                        os.path.basename(out_file).replace('.tar', ''), idx))
                    tar_info.size = len(js_serialized)
                    writer.addfile(tar_info, save_file)

                    # Write the vocab to file, if applicable
                    art_tokens = ' '.join(article_sents).split()
                    abs_tokens = ' '.join(abstract_sents).split()
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens] # strip
                    tokens = [t for t in tokens if t != ""] # remove empty
                    vocab_counter.update(tokens)
    print("Finished writing file {}\n".format(out_file))
    # write vocab to file
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
            'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)
    print("Finished writing vocab file")

