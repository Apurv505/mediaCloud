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


#We have main downloaded_stories folder and inside it each query creates a folder using the year. Then in each year we create a readme file 
# and put the actual query we sent to media cloud to keep track. Then we create new folder based on which media we are downloading stories from and put 
# the stories in there
 

import csv
import datetime
import time
import hashlib
import json
import multiprocessing
import os
from argparse import Namespace, ArgumentParser
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
import trafilatura


import socks
import socket

from utils import handle_dirs


def parse_args():
    parser = ArgumentParser(description="Download MediaCloud stories with language filter and rate limiting.")
    parser.add_argument('--language', type=str, default=None, help='Language code to filter stories (e.g., en, he, ar)')
    parser.add_argument('--rate_limit', type=float, default=None, help='Minimum seconds between API requests (rate limit)')
    parser.add_argument('--output_topic', type=str, default=None, help='Topic name for output directory (default: first query topic)')
    return parser.parse_args()

MAX_WORKERS = 1
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


def stories_about_topic(api_gen, mc, query, start_date, end_date, source_id, collection_ids, fetch_size=10, limit=10, language=None, rate_limit=None):
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
    normalized_lang = language.lower().strip() if language else None

    # Language filtering works reliably via search query syntax for story-list.
    base_query = f"({query}) AND language:{normalized_lang}" if normalized_lang else query

    while True:
        try:
            search_query = base_query
            source_filter = source_id
            if not use_source_ids:
                search_query = f"{base_query} AND media_id:{source_id}"
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

            # Defensive fallback in case backend ignores language clause for some sources.
            if normalized_lang:
                fetched_stories = [
                    s for s in fetched_stories
                    if s.get('language') and
                    str(s.get('language')).lower().startswith(normalized_lang)
                ]

            if rate_limit:
                time.sleep(rate_limit)
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
    Save a dict with MediaCloud data, website content or error, in downloaded_stories/{year}/
    """
    article = Article(story['url'])
    pub_date = story.get('publish_date')
    year = None
    if pub_date:
        try:
            year = str(pub_date)[:4]
        except Exception:
            year = 'unknown'
    else:
        year = 'unknown'

    # Get media name (safe for folder name)
    media_name = story.get('media_name') or story.get('media', 'unknown_media')
    # Clean media_name for filesystem
    import re
    media_name_clean = re.sub(r'[^\w\-_\. ]', '_', str(media_name))

    # Build a stable hash key from available fields in the new response.
    hash_obj = hashlib.blake2b(digest_size=20)
    hash_seed = f"{story.get('id', '')}|{story.get('url', '')}|{story.get('publish_date', '')}"
    hash_obj.update(hash_seed.encode('utf-8'))
    hashed_id = hash_obj.hexdigest()

    # Folder structure: downloaded_stories/{year}/readme.txt, then per-media subfolder, then stories inside media folder
    year_dir = os.path.join('downloaded_stories', year)
    handle_dirs(year_dir)

    # Write readme.txt with the query if not exists
    readme_path = os.path.join(year_dir, 'readme.txt')
    if not os.path.exists(readme_path):
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"Query: {cur_topic}\n")

    media_dir = os.path.join(year_dir, media_name_clean)
    handle_dirs(media_dir)
    json_file_name = os.path.join(media_dir, f"{hashed_id}.json")

    def convert_datetimes(obj):
        if isinstance(obj, dict):
            return {k: convert_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetimes(i) for i in obj]
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        else:
            return obj

    result = {
        "mediacloud_data": convert_datetimes(story),
        "website_content": None,
        "error": None
    }

    try:
        html = trafilatura.fetch_url(story['url'])
        text = None
        if html:
            text = trafilatura.extract(
                html,
                url=story['url'],
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )

        if text:
            result["website_content"] = fix_text(text).strip()
            status = 'success'
        else:
            article.download()
            article.parse()
            text = fix_text(article.text).strip()
            if text:
                result["website_content"] = text
                status = 'success'
            else:
                raise ValueError("No website text could be extracted")
    except Exception as e:
        result["error"] = str(e)
        status = 'fail'

    tmp_file_name = json_file_name + '.tmp'
    with open(tmp_file_name, 'w', encoding='utf-8') as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp_file_name, json_file_name)

    return Response(result, status)


def get_many_articles(cur_topics, stories, save_format='json', already_downloaded_hashes=None):
    responses = []
    counter = Counter()
    workers = min(MAX_WORKERS, len(stories))


    with futures.ThreadPoolExecutor(workers) as executor:
        to_do_map = {}
        for story in stories:
            # Build hash for this story as in get_one_article
            hash_obj = hashlib.blake2b(digest_size=20)
            hash_seed = f"{story.get('id', '')}|{story.get('url', '')}|{story.get('publish_date', '')}"
            hash_obj.update(hash_seed.encode('utf-8'))
            hashed_id = hash_obj.hexdigest()
            if already_downloaded_hashes and hashed_id in already_downloaded_hashes:
                counter['skipped'] += 1
                continue
            future = executor.submit(get_one_article, story, cur_topics, save_format)
            to_do_map[future] = story
        done_iter = futures.as_completed(to_do_map)

        for future in tqdm(done_iter, total=len(to_do_map), ascii=True):
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



def get_downloaded_hashes(output_dir, topic):
    topic_dir = os.path.join(output_dir, topic)
    if not os.path.exists(topic_dir):
        return set()
    hashes = set()
    for fname in os.listdir(topic_dir):
        if fname.endswith('.json'):
            hashes.add(fname.split('.')[0])
    return hashes


if __name__ == '__main__':


    cli_args = parse_args()

    # SET YOUR API KEYS IN THE TXT FILE !!!
    apis = get_list_of_APIs('api_key.txt')
    api_gen = (api for api in apis)
    mc = mediacloud.api.SearchApi(next(api_gen)[0])  # call the generator

    # SET YOUR QUERY TOPICS HERE !!!
    query_topics = ["merkel AND flucht*"]
    # collection_ids = ["34412234"]  # SET YOUR COLLECTION IDS HERE (OPTIONAL) !!!
    collection_ids = []  # SET YOUR COLLECTION IDS HERE (OPTIONAL) !!!
    # SET YOUR PERIOD HERE !!!
    start_date = datetime.date(2015, 1, 1)
    end_date = datetime.date(2015, 12, 30)

    # Use CLI topic for output dir if provided
    output_topic = cli_args.output_topic or query_topics[0]
    already_downloaded_hashes = get_downloaded_hashes('output_2021', output_topic)


    for topics in query_topics:
        for media in Media:
            cur_media_id = media.value
            query = topics
            res_stories, mc = stories_about_topic(
                api_gen,
                mc,
                query,
                start_date,
                end_date,
                cur_media_id,
                collection_ids=collection_ids,
                fetch_size=5,
                limit=100,
                language=cli_args.language,
                rate_limit=cli_args.rate_limit
            )
            print("We have fetched {} stories from {} about {}".format(len(res_stories), media.name, topics))
            if len(res_stories) != 0:
                story_responses = get_many_articles(
                    topics,
                    res_stories,
                    save_format='json',
                    already_downloaded_hashes=already_downloaded_hashes
                )
                print("Finished! {} success, {} failure, {} skipped".format(
                    story_responses.count['success'],
                    story_responses.count['fail'],
                    story_responses.count['skipped']
                ))
                print('*' * 40)
                print()
