#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmouth.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import codecs
import json
import os
import time
from tqdm import tqdm
import pandas as pd
import colorama
from collections import Counter
from multiprocessing import Pool, cpu_count

import nltk
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.api import CorpusReader

DOC_PATTERN = r'.*\.json'
CAT_PATTERN = r'([a-z_\s]+)/.*'



class MediaCloud_DataReader(CategorizedCorpusReader, CorpusReader):
    def __init__(self, corpus_root, fileids=DOC_PATTERN, encoding='utf-8', **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, corpus_root, fileids, encoding)

    def categories(self, fileids=None):
        # Override to avoid NLTK error when no categories exist (flat directory)
        return []

    def resolve(self, fileids=None, categories=None):
        """
        return file ids given explicit fileids or categories
        :param fileids:
        :param categories:
        :return:
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)  # return a list of identifiers (json path) in this corpus
        return fileids

        def docs(self, fileids=None, categories=None):
            """
            Returns the complete text of an HTML document, closing the document
            after we are done reading it and yielding it in a memory safe fashion.
            """
            # Resolve the fileids and the categories
            fileids = self.resolve(fileids, categories)

            # Create a generator, loading one document into memory at a time.
            # abspaths will return a list of all file identifiers in this corpus
            for path, encoding in self.abspaths(fileids, include_encoding=True):
                with codecs.open(path, 'r', encoding=encoding) as f:
                    yield json.load(f)  # will generate new dicts

        def sizes(self, fileids=None, categories=None):
            """
            Returns a list of tuples, the fileid and size on disk of the file.
            This function is used to detect oddly large files in the corpus.
            """
            # Resolve the fileids and the categories
            fileids = self.resolve(fileids, categories)

            # Create a generator, getting every path and computing filesize
            for path in self.abspaths(fileids):
                yield os.path.getsize(path)

        def get_title(self, fileids=None, categories=None):
            """
            Return the title from mediacloud_data
            """
            for doc in self.docs(fileids, categories):
                yield doc.get('mediacloud_data', {}).get('title', None)

        def get_media(self, fileids=None, categories=None):
            """
            Return the media name from mediacloud_data
            """
            for doc in self.docs(fileids, categories):
                yield doc.get('mediacloud_data', {}).get('media', None)

        def get_author(self, fileids=None, categories=None):
            """
            Return the author from mediacloud_data
            """
            for doc in self.docs(fileids, categories):
                yield doc.get('mediacloud_data', {}).get('author', None)

        def get_media_label(self, fileids=None, categories=None):
            """
            Return the media label based on media name from mediacloud_data
            """
            for doc in self.docs(fileids, categories):
                me = doc.get('mediacloud_data', {}).get('media', None)
                if me in ['BBC', 'CNN', 'New York Times', 'NPR', 'Washington Post', 'HuffPost', 'guardiannews.com']:
                    yield 0
                elif me in ['CNBC', 'USA Today', 'Wall Street Journal', 'CBS News', 'ABC.com']:
                    yield 1
                elif me in ['rushlimbaugh.com', 'The Sean Hannity Show', 'Fox News', 'Breitbart']:
                    yield 2
                else:
                    print(me)
                    yield 2

        def get_pubdate(self, fileids=None, categories=None):
            """
            Return the publication date from mediacloud_data
            """
            for doc in self.docs(fileids, categories):
                yield doc.get('mediacloud_data', {}).get('pub_date', None)

        def get_keywords(self, fileids=None, categories=None):
            """
            Return the keywords from mediacloud_data
            """
            for doc in self.docs(fileids, categories):
                yield doc.get('mediacloud_data', {}).get('keywords', None)

        def clean_text(self, fileids=None, categories=None):
            """
            Returns the website content from website_content field.
            """
            for doc in self.docs(fileids, categories):
                yield doc.get('website_content', None)

        def paras(self, fileids=None, categories=None):
            """
            Uses BeautifulSoup to parse the paragraphs from the HTML.
            """
            for text in self.clean_text(fileids, categories):
                for para in text.split('\n\n'):
                    yield para

        def sents(self, fileids=None, categories=None):
            """
            Uses the built in sentence tokenizer to extract sentences from the
            paragraphs. Note that this method uses BeautifulSoup to parse HTML.
            """
            for paragraph in self.paras(fileids, categories):
                for sentence in sent_tokenize(paragraph):
                    yield sentence

        def words(self, fileids=None, categories=None):
            """
            Uses the built in word tokenizer to extract tokens from sentences.
            Note that this method uses BeautifulSoup to parse HTML content.
            """
            for sentence in self.sents(fileids, categories):
                for token in wordpunct_tokenize(sentence):
                    yield token

        def tokenize(self, fileids=None, categories=None):
            """
            Segments, tokenizes, and tags a document in the corpus.
            """
            for paragraph in self.paras(fileids=fileids):
                yield [
                    pos_tag(wordpunct_tokenize(sent))
                    for sent in sent_tokenize(paragraph)
                ]

        def __len__(self, fileids=None, categories=None):
            return len(self.resolve(fileids, categories) or self.fileids())

        def show_stats(self, fileids=None, categories=None):
            """
            Performs a single pass of the corpus and
            returns a dictionary with a variety of metrics
            concerning the state of the corpus.
            """
            started = time.time()

            # Structures to perform counting.
            counts = nltk.FreqDist()
            tokens = nltk.FreqDist()

            # Perform single pass over paragraphs, tokenize_pos and count
            for para in self.paras(fileids, categories):
                counts['paras'] += 1

                for sent in sent_tokenize(para):
                    counts['sents'] += 1

                    for word in wordpunct_tokenize(sent):
                        counts['words'] += 1
                        tokens[word] += 1

            # Compute the number of files and categories in the corpus
            n_fileids = len(self.resolve(fileids, categories) or self.fileids())
            n_topics = len(self.categories(self.resolve(fileids, categories)))

            # Return data structure with information
            print(
                "files: {}, topics: {}, paras: {}, sents: {}, words: {}, vocab: {},"
                " lexdiv: {:0.2f}, ppdoc: {:0.2f}, sppar: {:0.2f}, secs: {:0.2f}".format(
                    n_fileids, n_topics, counts['paras'],
                    counts['sents'], counts['words'],
                    len(tokens), float(counts['words']) / float(len(tokens)), float(counts['paras']) / float(n_fileids),
                    float(counts['sents']) / float(counts['paras']), time.time() - started))

    def resolve(self, fileids=None, categories=None):
        """
        return file ids given explicit fileids or categories
        :param fileids:
        :param categories:
        :return:
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)  # return a list of identifiers (json path) in this corpus
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete text of an HTML document, closing the document
        after we are done reading it and yielding it in a memory safe fashion.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        # abspaths will return a list of all file identifiers in this corpus
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield json.load(f)  # will generate new dicts

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)


    def get_title(self, fileids=None, categories=None):
        """
        Return the title from mediacloud_data
        """
        for doc in self.docs(fileids, categories):
            yield doc.get('mediacloud_data', {}).get('title', None)


    def get_media(self, fileids=None, categories=None):
        """
        Return the media name from mediacloud_data
        """
        for doc in self.docs(fileids, categories):
            yield doc.get('mediacloud_data', {}).get('media', None)


    def get_author(self, fileids=None, categories=None):
        """
        Return the author from mediacloud_data
        """
        for doc in self.docs(fileids, categories):
            yield doc.get('mediacloud_data', {}).get('author', None)


    def get_media_label(self, fileids=None, categories=None):
        """
        Return the media label based on media name from mediacloud_data
        """
        for doc in self.docs(fileids, categories):
            me = doc.get('mediacloud_data', {}).get('media', None)
            if me in ['BBC', 'CNN', 'New York Times', 'NPR', 'Washington Post', 'HuffPost', 'guardiannews.com']:
                yield 0
            elif me in ['CNBC', 'USA Today', 'Wall Street Journal', 'CBS News', 'ABC.com']:
                yield 1
            elif me in ['rushlimbaugh.com', 'The Sean Hannity Show', 'Fox News', 'Breitbart']:
                yield 2
            else:
                print(me)
                yield 2


    def get_pubdate(self, fileids=None, categories=None):
        """
        Return the publication date from mediacloud_data
        """
        for doc in self.docs(fileids, categories):
            yield doc.get('mediacloud_data', {}).get('pub_date', None)


    def get_keywords(self, fileids=None, categories=None):
        """
        Return the keywords from mediacloud_data
        """
        for doc in self.docs(fileids, categories):
            yield doc.get('mediacloud_data', {}).get('keywords', None)


    def clean_text(self, fileids=None, categories=None):
        """
        Returns the website content from website_content field.
        """
        for doc in self.docs(fileids, categories):
            yield doc.get('website_content', None)

    def paras(self, fileids=None, categories=None):
        """
        Uses BeautifulSoup to parse the paragraphs from the HTML.
        """
        for text in self.clean_text(fileids, categories):
            for para in text.split('\n\n'):
                yield para

    def sents(self, fileids=None, categories=None):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        paragraphs. Note that this method uses BeautifulSoup to parse HTML.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Uses the built in word tokenizer to extract tokens from sentences.
        Note that this method uses BeautifulSoup to parse HTML content.
        """
        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids=None, categories=None):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def __len__(self, fileids=None, categories=None):
        return len(self.resolve(fileids, categories) or self.fileids())

    def show_stats(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        started = time.time()

        # Structures to perform counting.
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize_pos and count
        for para in self.paras(fileids, categories):
            counts['paras'] += 1

            for sent in sent_tokenize(para):
                counts['sents'] += 1

                for word in wordpunct_tokenize(sent):
                    counts['words'] += 1
                    tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        print(
            "files: {}, topics: {}, paras: {}, sents: {}, words: {}, vocab: {},"
            " lexdiv: {:0.2f}, ppdoc: {:0.2f}, sppar: {:0.2f}, secs: {:0.2f}".format(
                n_fileids, n_topics, counts['paras'],
                counts['sents'], counts['words'],
                len(tokens), float(counts['words']) / float(len(tokens)), float(counts['paras']) / float(n_fileids),
                float(counts['sents']) / float(counts['paras']), time.time() - started))


class Preprocessor(object):
    def __init__(self, corpus, target=None, **kwargs):
        """
        convert the corpus to dataframe, with corresponding attributes filled
        :param corpus:
        :param out_name:
        :param target:
        :param kwargs:
        """
        self.corpus = corpus
        self.target = target

    def get_fileids_size(self, fileids=None, categories=None):
        return len(self.corpus.resolve(fileids, categories))

    def get_fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def tokenize_pos(self, fileid):
        """
        returns a generator of paragraphs, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        for paragraph in self.corpus.paras(fileids=fileid):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def plain_text(self, fileid):
        """
        return list of paragraphs for each article
        :param fileid:
        :return:
        """
        yield [para for para in self.corpus.paras(fileids=fileid)]

    def process(self, fileid):
        """
        single file processing function (given file id)
        :param fileid:
        :return: a dict with attributes filled in, or None if error/empty
        """
        print(f"Processing file: {fileid}")
        with codecs.open(fileid, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        if doc.get('error'):
            print(f"  Skipped: error present: {doc.get('error')}")
            return None
        if not doc.get('website_content'):
            print(f"  Skipped: website_content is empty or missing")
            return None
        mc = doc.get('mediacloud_data', {})
        document = {
            'title': mc.get('title'),
            'author': mc.get('author'),
            'media': mc.get('media_name') or mc.get('media'),
            'media_label': None,  # will fill below
            'pubdate': mc.get('publish_date'),
            'words': doc.get('website_content'),
        }
        print(f"  Extracted fields: title={document['title']}, author={document['author']}, media={document['media']}, pubdate={document['pubdate']}")
        liberal = ['BBC', 'CNN', 'New York Times', 'NPR', 'Washington Post', 'HuffPost', 'guardiannews.com']
        neutral = ['CNBC', 'USA Today', 'Wall Street Journal', 'CBS News', 'ABC.com']
        conservative = ['rushlimbaugh.com', 'The Sean Hannity Show', 'Fox News', 'Breitbart']
        me = document['media']
        if me in liberal:
            document['media_label'] = 0
        elif me in neutral:
            document['media_label'] = 1
        elif me in conservative:
            document['media_label'] = 2
        else:
            document['media_label'] = 2
        print(f"  Assigned media_label: {document['media_label']}")
        return document

    def transform(self, fileids=None, categories=None, thread_num=2):
        """
        multi-thread transforming
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        def save_single_result(args):
            fileid, doc = args
            if doc:
                base = os.path.splitext(os.path.basename(fileid))[0]
                out_path = os.path.join(self.target, base + '.csv')
                df = pd.DataFrame([doc])
                df.to_csv(out_path, index=False)

        if fileids is not None and categories is None:
            print("Processing all provided fileids as a single batch.")
            with Pool(thread_num) as proc:
                results = list(tqdm(proc.imap(self.process, fileids), total=len(fileids)))
            for fileid, doc in zip(fileids, results):
                save_single_result((fileid, doc))
        else:
            with Pool(thread_num) as proc:
                for cate in categories:
                    print("preprocessing topic:", cate)
                    fileid_list = self.get_fileids(fileids, cate)
                    results = list(
                        tqdm(proc.imap(self.process, fileid_list),
                             total=self.get_fileids_size(categories=cate)))
                    for fileid, doc in zip(fileid_list, results):
                        save_single_result((fileid, doc))
                    print()


if __name__ == '__main__':
    colorama.init(autoreset=True)

    # picked_category = ['drones', 'abortion']  # if you want all topics, set to None
    picked_category = None
    processes_num = 20
    processes = min(processes_num, cpu_count())

    print()

    for year in ['2026']:
        print(colorama.Fore.LIGHTBLUE_EX + "= Loading the dataset ... " + "Year: " + year)
        corpus = MediaCloud_DataReader('./downloaded_stories/' + year)

        category = picked_category if picked_category else corpus.categories()
        print(f"Categories found: {category}")
        # Use full paths for fileids
        all_fileids = [os.path.join('./downloaded_stories', year, fname) for fname in os.listdir(os.path.join('./downloaded_stories', year)) if fname.endswith('.json')]
        print(f"Fileids found: {all_fileids}")
        print()
        print(colorama.Fore.LIGHTBLUE_EX + "= Showing the dataset statistics:")
        # corpus.show_stats(categories=category)
        print()

        since = time.time()
        preprocessor = Preprocessor(corpus, './csv_output_' + year + '/')
        print(colorama.Fore.LIGHTBLUE_EX + "= Preprocessing ...")
        # If no categories, process all files directly
        if not category:
            print("No categories found, processing all files directly.")
            preprocessor.transform(thread_num=processes, fileids=all_fileids, categories=None)
        else:
            preprocessor.transform(thread_num=processes, categories=category)
        print()

        time_elapsed = time.time() - since
        print(colorama.Fore.LIGHTGREEN_EX + "= Preprocessing is done!")
        print('It takes {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
