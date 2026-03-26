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


import csv
import datetime
import hashlib
import json
import multiprocessing
import os
from argparse import Namespace
from collections import namedtuple, Counter
from concurrent import futures
import mediacloud.api
import mediacloud.error
import newspaper
import requests
from ftfy import fix_text
from newspaper import Article
from tqdm import tqdm
from media_list import Media

import socks
import socket

from utils import handle_dirs

MAX_WORKERS = 1000
MAX_CPUS = multiprocessing.cpu_count()

Story = namedtuple('Story', ['id',
                             'title',
                             'author',
                             'media',
                             'media_url',
                             'story_url',
                             'pub_date',
                             'stories_id',
                             'guid',
                             'processed_stories_id',
                             'text'])

Response = namedtuple('Response', ['response', 'status'])
Responses = namedtuple('Responses', ['responses', 'count'])


args = Namespace(
    load_dir='./',
    save_dir='./output_test/',
    output_dir='./output_new/',
    opml_file_name='feedly_new.opml',
    rss_format='xml',
    output_format='html',
    json_file_name='opml.json',
    expand_filepaths_to_save_dir=True,
)


def get_list_of_APIs(path):
    apis = []
    with open(path, 'r') as f:
        for line in f:
            apis.append(list(line.strip('\n').split(',')))
    return apis


def save_as_csv(save_dir, csv_file_name, content):
    """
    Save the content to a csv file
    :param save_dir: the saving directory
    :param csv_file_name: hashed id
    :param content: (namedtuple Story)
    :return:
    """
    handle_dirs(save_dir)
    csv_file_path = os.path.join(save_dir, csv_file_name)
    with open(csv_file_path, 'w+') as fp:
        f_csv = csv.DictWriter(fp, content._fields)
        f_csv.writeheader()
        f_csv.writerow(content._asdict())


def save_as_json(save_dir, json_file_name, content):
    """
    Save the content to a json file
    :param save_dir: the saving directory
    :param json_file_name: the json file name
    :param content: the content to be saved
    :return:
    """
    handle_dirs(save_dir)
    json_file_path = os.path.join(save_dir, json_file_name)
    with open(json_file_path, 'w+', encoding='utf-8') as fp:
        # Convert date/datetime values (eg publish_date) to ISO strings.
        fp.write(json.dumps(content._asdict(), indent=2, default=str))
    # print("Save as json successfully!")


def save_as_txt(save_dir, txt_file_name, content):
    """
    Save the content to a json file
    :param save_dir: the saving directory
    :param txt_file_name: the txt file name
    :param content: the content to be saved
    :return:
    """
    handle_dirs(save_dir)
    txt_file_path = os.path.join(save_dir, txt_file_name)
    fp = open(txt_file_path, 'w+')
    fp.write(json.dumps(content, indent=2))
    fp.close()
    # print("Save as txt successfully!")


def set_themes(stories):
    """
    set the theme attr for each story
    :param stories:
    :return:
    """
    for s in stories:
        # Tags are not returned by the new API; preserve key for downstream compatibility.
        s['themes'] = ''
    return stories


def story_list_compat(mc, query, start_date, end_date, source_id=None, collection_ids=None, pagination_token=None, page_size=10):
    """Compatibility wrapper for client versions that build tuple params for `ss`/`cs`."""
    source_ids = [source_id] if source_id is not None else []
    params = mc._prep_default_params(query, start_date, end_date, source_ids=source_ids, collection_ids=collection_ids)
    if isinstance(params.get('ss'), tuple):
        params['ss'] = params['ss'][0]
    if isinstance(params.get('cs'), tuple):
        params['cs'] = params['cs'][0]
    if pagination_token:
        params['pagination_token'] = pagination_token
    if page_size:
        params['page_size'] = page_size

    # Use a local request flow so non-JSON error pages don't crash with JSONDecodeError.
    endpoint_url = mc.BASE_API_URL + 'search/story-list'
    response = mc._session.get(endpoint_url, params=params, timeout=mc.TIMEOUT_SECS)
    results = None
    parse_error = None
    try:
        results = response.json()
    except ValueError as exc:
        parse_error = exc

    if response.status_code != 200:
        # Preserve HTTP error semantics (especially 429) even when the body is empty/non-JSON.
        error_data = results if isinstance(results, dict) else {}
        raise mediacloud.error.APIResponseError(response, params, error_data)

    if parse_error is not None:
        body_preview = response.text[:200].replace('\n', ' ')
        raise RuntimeError(
            f"MediaCloud returned non-JSON response for story-list (status={response.status_code}). "
            f"Body preview: {body_preview!r}"
        ) from parse_error

    if not isinstance(results, dict) or 'stories' not in results:
        raise RuntimeError("MediaCloud story-list response missing expected 'stories' field")

    mc._dates_str2objects(results['stories'])
    return results['stories'], results.get('pagination_token')


def stories_about_topic(api_gen, mc, query, start_date, end_date, source_id, collection_ids, fetch_size=10, limit=10):
    """
    Return stories on certain topic from certain source, from start_time to end_time.
    :param mc: the media cloud client
    :param query: the query string
    :param start_date: requested start date
    :param end_date: requested end date
    :param source_id: media source id to filter stories
    :param collection_ids: collection ids to filter stories
    :param fetch_size:
    :param limit: max number of return stories
    :return: a tuple of (list of stories, active media cloud client)
    """

    stories = []
    pagination_token = None
    use_source_ids = True

    while True:
        try:
            search_query = query
            source_filter = source_id
            if not use_source_ids:
                search_query = f"{query} AND media_id:{source_id}"
                source_filter = None

            fetched_stories, pagination_token = story_list_compat(
                mc,
                search_query,
                start_date,
                end_date,
                source_filter,
                collection_ids=collection_ids,
                pagination_token=pagination_token,
                page_size=fetch_size,
            )
        except mediacloud.error.APIResponseError as e:
            if e.response.status_code == 429:
                print()
                print("Switch media cloud account!")
                print()
                try:
                    mc = mediacloud.api.SearchApi(next(api_gen)[0])  # call the generator
                except StopIteration as exc:
                    raise RuntimeError("All API keys exhausted while handling 429 responses") from exc
                continue

            error_note = ''
            if isinstance(e.data, dict):
                error_note = str(e.data.get('note', ''))
            if e.response.status_code == 422 and use_source_ids and 'No sources found' in error_note:
                print("Source id filter rejected by API; falling back to legacy media_id query filter.")
                use_source_ids = False
                pagination_token = None
                stories = []
                continue

            raise

        if len(fetched_stories) == 0:
            break

        stories += fetched_stories
        if len(stories) >= limit:
            stories = stories[:limit]
            break

        if pagination_token is None:
            break

    stories = set_themes(stories)
    return stories, mc


def get_one_article(story, cur_topic, save_format='json'):
    """
    Return a dict that stores all the information extracted from url
    :param cur_topic: current query topic
    :param save_format: 'json' or 'txt', as file format
    :param story: (story) a object from media cloud
    :return: the text of the story
    """
    response = Response
    article = Article(story['url'])

    if not article.is_media_news():
        try:
            article.download()
            article.parse()
        except newspaper.ArticleException:
            status = "fail"
            return Response(response, status)
        else:
            text = fix_text(article.text)

            # if no exception, set status to success
            status = 'success'

            # set attributes that story already has
            title = story['title']
            media_name = story['media_name']
            media_url = story['media_url']
            story_url = story.get('url')
            pub_date = story['publish_date']
            # The new API returns `id` and no legacy guid/processed id fields.
            stories_id = story.get('id')
            guid = None
            processed_stories_id = None

            # Build a stable hash key from available fields in the new response.
            hash_obj = hashlib.blake2b(digest_size=20)
            hash_seed = f"{story.get('id', '')}|{story.get('url', '')}|{story.get('publish_date', '')}"
            hash_obj.update(hash_seed.encode('utf-8'))
            hashed_id = hash_obj.hexdigest()

            # get authors from the story with newspaper
            author = article.authors

            response = Story(hashed_id, title, author, media_name, media_url, story_url, pub_date, stories_id, guid,
                             processed_stories_id, text)
    else:
        status = 'fail'
        return Response(response, status)

    if save_format == 'txt':
        txt_file_name = ''.join([hashed_id, '.txt'])
        save_as_txt(''.join(['output_2021/', cur_topic]), txt_file_name, text)
    elif save_format == 'json':
        json_file_name = ''.join([hashed_id, '.json'])
        save_as_json(''.join(['output_2021/', cur_topic]), json_file_name, response)
    elif save_format == 'csv':
        csv_file_name = ''.join([hashed_id, '.csv'])
        save_as_csv(''.join(['output_2021/', cur_topic]), csv_file_name, response)

    return Response(response, status)


def get_many_articles(cur_topics, stories, save_format='json'):
    responses = []
    counter = Counter()
    workers = min(MAX_WORKERS, len(stories))

    with futures.ThreadPoolExecutor(workers) as executor:
        to_do_map = {}
        for story in stories:
            future = executor.submit(get_one_article, story, cur_topics, save_format)
            to_do_map[future] = story
        done_iter = futures.as_completed(to_do_map)

        for future in tqdm(done_iter, total=len(stories), ascii=True):
            try:
                res = future.result()
            except newspaper.ArticleException as article_exc:
                print(article_exc)
                get_many_status = 'fail'
            except requests.exceptions.HTTPError as exc:
                get_many_status = 'fail'
                error_msg = 'HTTP error {res.status_code} - {res.reason}'
                error_msg = error_msg.format(res=exc.response)
                print(error_msg)
            except requests.exceptions.ConnectionError as exc:
                get_many_status = 'fail'
                print('Connection error')
            else:
                get_many_status = res.status
                responses.append(res.response)

            counter[get_many_status] += 1

    return Responses(responses, counter)


if __name__ == '__main__':

    # SET YOUR API KEYS IN THE TXT FILE !!!
    apis = get_list_of_APIs('api_key.txt')
    api_gen = (api for api in apis)
    mc = mediacloud.api.SearchApi(next(api_gen)[0])  # call the generator

    # SET YOUR QUERY TOPICS HERE !!!
    query_topics = ["israel"]
    collection_ids = ["34412234"]  # SET YOUR COLLECTION IDS HERE (OPTIONAL) !!!
    # SET YOUR PERIOD HERE !!!
    start_date = datetime.date(2026, 1, 1)
    end_date = datetime.date(2026, 1, 31)

    for topics in query_topics:
        for media in Media:
            cur_media_id = media.value
            query = topics
            res_stories, mc = stories_about_topic(api_gen,
                                                  mc,
                                                  query,
                                                  start_date,
                                                  end_date,
                                                  cur_media_id,
                                                  collection_ids=collection_ids,
                                                  fetch_size=50,
                                                  limit=50)
            print("We have fetched {} stories from {} about {}".format(len(res_stories), media.name, topics))
            if len(res_stories) != 0:
                story_responses = get_many_articles(topics, res_stories, save_format='json')
                print("Finished! {} success, and {} failure".format(story_responses.count['success'],
                                                                    story_responses.count['fail']))
                print('*' * 40)
                print()
