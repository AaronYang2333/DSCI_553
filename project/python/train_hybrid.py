__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '4/29/2020 11:21 PM'

import collections
import itertools
import json
import random
import sys
import os
import time
from math import sqrt, ceil

from pyspark import SparkConf, SparkContext

USER_ID = 'user_id'
BUSINESS_ID = 'business_id'
SCORE = "stars"
FRIENDS = "friends"
REVIEW_COUNT = "review_count"
USEFUL = "useful"
CO_RATED_THRESHOLD = 3
NUM_OF_HASH_FUNC = 300
BANDS = 200
N_MOST_POWERFUL = 10
JACCARD_SIMILARITY_THRESHOLD = 0.01

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


def genHashFuncs(num_of_func, baskets):
    """
    generate a list of hash funcs
    :param num_of_func: the number of hash func you need
    :param baskets: the number of baskets a hash func should use
    :return: a list of func object
    """
    func_list = list()

    def build_func(param_a, param_b, param_m):
        def apply_funcs(input_x):
            return ((param_a * input_x + param_b) % sys.maxsize) % param_m

        return apply_funcs

    param_as = random.sample(range(1, sys.maxsize - 1), num_of_func)
    param_bs = random.sample(range(0, sys.maxsize - 1), num_of_func)
    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b, baskets))

    return func_list


def computeSimilarity(dict1, dict2):
    """
    compute Pearson Correlation Similarity
    :param dict1:
    :param dict2:
    :return: a float number
    """
    co_rated_user = list(set(dict1.keys()) & (set(dict2.keys())))
    val1_list, val2_list = list(), list()
    [(val1_list.append(dict1[user_id]),
      val2_list.append(dict2[user_id])) for user_id in co_rated_user]

    avg1 = sum(val1_list) / len(val1_list)
    avg2 = sum(val2_list) / len(val2_list)

    numerator = sum(map(lambda pair: (pair[0] - avg1) * (pair[1] - avg2),
                        zip(val1_list, val2_list)))

    if numerator == 0:
        return 0
    denominator = sqrt(sum(map(lambda val: (val - avg1) ** 2, val1_list))) * \
                  sqrt(sum(map(lambda val: (val - avg2) ** 2, val2_list)))
    if denominator == 0:
        return 0

    return numerator / denominator


def export2File(data_type, data, file_path):
    """
    export json content to a file
    :param data_type: array, dict, or sth else
    :param data: a list of dict
    :param file_path: output file path
    :return: nothing, but a file
    """
    if data_type == "array":
        with open(file_path, 'w+') as output_file:
            for item in data:
                output_file.writelines(json.dumps(item) + "\n")
    else:
        with open(file_path, 'w+') as output_file:
            pairs = json.dumps(data)
            output_file.write(pairs)


def flatMixedList(dict_list):
    """
    flat the dict_list into a big dict
    :param dict_list: [{a: 1}, {b: 2}, {c: 3}, ...]
    :return: a dict {a:!, b:2, c:3,...}
    """
    result = collections.defaultdict(list)
    for item in dict_list:
        result[list(item.keys())[0]] = list(item.values())[0]
    return result


def applyHashFuncs(funcs, index):
    """
    apply hash func on index number
    :param funcs:
    :param index:
    :return:
    """
    return list(map(lambda func: func(index), funcs))


def getMinValue(list1, list2):
    """
    get min value in each element in two list
    :param list1: e.g. [3,6,2,6,8]
    :param list2: e.g. [1,4,5,6,2]
    :return: a list which contain the min value in each column
        e.g.  =======> [1,4,2,6,2]
    """
    return [min(val1, val2) for val1, val2 in zip(list1, list2)]


def splitList(value_list, chunk_num):
    """
    split a list in to several chunks
    :param value_list: a list whose shape is [N]
    :param chunk_num: the number of chunk you want to split
    :return: a list of list
    e.g. return [[1,a], [2,b], [3,c], [4,d]] and a + b + c + d = N
    """
    chunk_lists = list()
    size = int(ceil(len(value_list) / int(chunk_num)))
    for index, start_index in enumerate(range(0, len(value_list), size)):
        chunk_lists.append((index, hash(tuple(value_list[start_index:start_index + size]))))
    return chunk_lists


def computeJaccard(dict1, dict2):
    """
    compute Jaccard Similarity
    :param dict1:
    :param dict2:
    :return: a float number
    """
    if dict1 is not None and dict2 is not None:
        users1 = set(dict1.keys())
        users2 = set(dict2.keys())
        if len(users1 & users2) >= CO_RATED_THRESHOLD:
            if float(float(len(users1 & users2)) / float(len(users1 | users2))) \
                    >= JACCARD_SIMILARITY_THRESHOLD:
                return True

    return False


def buildUserBaseCF(original_rdd, num_of_business, export_model_file_path):
    """
    build user based collaborative filtering
    :param original_rdd:
    :param num_of_business:
    :param export_model_file_path:
    :return:
    """
    hash_funcs = genHashFuncs(NUM_OF_HASH_FUNC, num_of_business * 2)

    # group original resource by bidx, and remove those unpopular business (rated time < 3)
    # tuple(bidx, (uidx, score))
    # [(5306, [(3662, 5.0), (3218, 5.0), (300, 5.0),..]), ()
    shrunk_bid_u_info_rdd = original_rdd \
        .map(lambda kv: (bus_index_dict[kv[1]], (user_index_dict[kv[0]], kv[2]))) \
        .groupByKey().mapValues(lambda uid_score: list(uid_score)) \
        .filter(lambda bid_uid_score: len(bid_uid_score[1]) >= CO_RATED_THRESHOLD) \
        .persist()

    # build min hash signature for every business index (bus_idx)
    # and generate user_index pair
    # tuple(uidx1, uidx2)
    # => [(24752, [0, 0, 0, 8, 0, 0,...]), (1666, [0, 0, 8, 3,...
    # => [(3218, 4128), (300, 3662), (300, 4128), (3218, 3662), (30 ...
    uidx_pair = shrunk_bid_u_info_rdd \
        .flatMap(lambda bid_uid_score: [(uid_score[0],
                                         applyHashFuncs(hash_funcs, bid_uid_score[0]))
                                        for uid_score in bid_uid_score[1]]) \
        .reduceByKey(getMinValue) \
        .flatMap(lambda kv: [(tuple(chunk), kv[0]) for chunk in splitList(kv[1], BANDS)]) \
        .groupByKey().map(lambda kv: sorted(set(kv[1]))).filter(lambda val: len(val) > 1) \
        .flatMap(lambda uid_list: [pair for pair in itertools.combinations(uid_list, 2)]) \
        .distinct()

    # convert shrunk_bid_uids_rdd into dict form
    # dict(uidx: dict{bidx:score,....})
    # => {2275: defaultdict(<class 'list'>, {4978: 4.0, 1025: 4.0, 545: 5.0,...
    uid_bids_dict = shrunk_bid_u_info_rdd \
        .flatMap(lambda bid_uid_score: [(item[0], (bid_uid_score[0], item[1]))
                                        for item in bid_uid_score[1]]) \
        .groupByKey().mapValues(lambda val: list(set(val))) \
        .filter(lambda uidx_mixed: len(uidx_mixed[1]) >= CO_RATED_THRESHOLD) \
        .mapValues(lambda vals: [{bid_score[0]: bid_score[1]} for bid_score in vals]) \
        .mapValues(lambda val: flatMixedList(val)) \
        .map(lambda uid_bid_score: {uid_bid_score[0]: uid_bid_score[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    # generate all possible pair between candidate uidx
    # and compute the pearson similarity
    candidate_pair = uidx_pair \
        .filter(lambda pair: computeJaccard(uid_bids_dict.get(pair[0], None),
                                            uid_bids_dict.get(pair[1], None))) \
        .map(lambda id_pair: (id_pair, computeSimilarity(uid_bids_dict[id_pair[0]],
                                                         uid_bids_dict[id_pair[1]]))) \
        .filter(lambda kv: kv[1] > 0) \
        .map(lambda kv: {"u1": reversed_index_user_dict[kv[0][0]],
                         "u2": reversed_index_user_dict[kv[0][1]],
                         "sim": kv[1]})

    export2File("array", candidate_pair.collect(), export_model_file_path)


def buildContentBaseModel(original_rdd, export_model_file_path):
    """
    build content based model
    :param original_rdd:
    :param export_model_file_path:
    :return:
    """
    # [('4N-HU_T32hLENLntsNKNBg', 56848), (' pSY2vwWLgWfGVAAiKQzMng', 56848)....
    # [('V7XFwm0baX37HRIduHmrXw', [(56848, 28), (44700, 22757), (69533, 10),
    fans_pair = original_rdd \
        .flatMap(lambda kk_v: [(friend.strip(), (user_index_dict.get(kk_v[0][0], -1), kk_v[0][1]))
                               for friend in kk_v[1]]) \
        .groupByKey().mapValues(list).filter(lambda kv: len(kv[1]) > 1) \
        .mapValues(lambda uid_score: sorted(uid_score,
                                            key=lambda item: item[1],
                                            reverse=True)[:N_MOST_POWERFUL]) \
        .mapValues(lambda uid_score: [reversed_index_user_dict.get(item[0], "UNK") for item in uid_score]) \
        .map(lambda kv: {"target": kv[0], "idol": kv[1]})

    export2File("array", fans_pair.collect(), export_model_file_path)


def buildAvgBaseModel(model_type, original_rdd, given_avg_dict, export_model_file_path):
    """
    build baseline model
    :param model_type:
    :param original_rdd:
    :param given_avg_dict:
    :param export_model_file_path:
    :return:
    """
    generated_dict = None
    if model_type == "USER":
        generated_dict = original_rdd.map(lambda kvv: (kvv[0], kvv[2])) \
            .groupByKey().mapValues(lambda val: sum(val) / len(val)).collectAsMap()
    else:
        generated_dict = original_rdd.map(lambda kvv: (kvv[1], kvv[2])) \
            .groupByKey().mapValues(lambda val: sum(val) / len(val)).collectAsMap()

    generated_dict.update(given_avg_dict)
    # export this combined dict
    export2File("dict", generated_dict, export_model_file_path)


if __name__ == '__main__':
    start = time.time()
    train_file_path = "../resource/asnlib/publicdata/train_review.json"
    user_graph_data_file_path = "../resource/asnlib/publicdata/user.json"

    given_bus_avg_file_path = "../resource/asnlib/publicdata/business_avg.json"
    given_user_avg_file_path = "../resource/asnlib/publicdata/user_avg.json"

    export_cf_model_file_path = "./user-based.model"
    export_content_model_file_path = "./content-based.model"
    export_user_avg_model_file_path = "./user-avg.model"
    export_bus_avg_model_file_path = "./bus-avg.model"

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("ay_inf_553_project_train") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    # ========================== Read Training Data ===============================
    # read training data and remove useless columns
    # tuple(uid, bid, score)
    # => ("OLR4DvqFxCKLOEHqfAxpqQ", "zK7sltLeRRioqYwgLiWUIA", 5.0) ...
    train_input_lines = sc.textFile(train_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID], kv[SCORE])) \
        .persist()

    # ========================== Read User JSON Data ===============================
    # read user.json to build friend circle
    # tuple((uid, useful_score), list)
    # => ((bc8C_eETBWL0olvFSJJd0w, 28), [xxx, xxx, xxx,...])
    user_graph_input_lines = sc.textFile(user_graph_data_file_path) \
        .map(lambda row: json.loads(row)) \
        .map(lambda kv: ((kv[USER_ID], kv[USEFUL]), kv[FRIENDS].split(','))) \
        .persist()

    # ========================== Read AVG Info Dict ===============================
    # read avg info from json file and convert it into dict
    # dict(uid_str: avg_score)
    # => {"MHiKdBFx4McRQONnuMbByw": 3.857142857142857, ...}
    user_avg_dict = sc.textFile(given_user_avg_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: dict(kv)).flatMap(lambda kv_items: kv_items.items()) \
        .collectAsMap()

    # read avg info from json file and convert it into dict
    # dict(bid_str: avg_score)
    # => {"AtD6B83S4Mbmq0t7iDnUVA": 4.393401015228426, ...}
    bus_avg_dict = sc.textFile(given_bus_avg_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: dict(kv)).flatMap(lambda kv_items: kv_items.items()) \
        .collectAsMap()

    # ========================== Build ID-Index Dict ===============================
    # collect (sorted & distinct) user and tokenize them
    # => generate dict(distinct user id: index(uidx))
    # => e.g. {'-2QGc6Lb0R027lz0DpWN1A': 1, 'xxx': int, ...} user count: 26184
    user_index_dict = train_input_lines.map(lambda kvv: kvv[0]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_user_dict = {v: k for k, v in user_index_dict.items()}

    # collect (sorted & distinct) business index and tokenize them
    # => generate dict(distinct business_id: index(bidx))
    # => e.g. {'--9e1ONYQuAa-CB_Rrw7Tw': 0, 'xxx': int, ....} business count: 10253
    bus_index_dict = train_input_lines.map(lambda kvv: kvv[1]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_bus_dict = {v: k for k, v in bus_index_dict.items()}

    # ========================== Build Different Model ===============================
    buildUserBaseCF(original_rdd=train_input_lines,
                    num_of_business=len(bus_index_dict),
                    export_model_file_path=export_cf_model_file_path)

    buildAvgBaseModel(model_type="USER", original_rdd=train_input_lines,
                      given_avg_dict=user_avg_dict,
                      export_model_file_path=export_user_avg_model_file_path)

    buildAvgBaseModel(model_type="BUSINESS", original_rdd=train_input_lines,
                      given_avg_dict=bus_avg_dict,
                      export_model_file_path=export_bus_avg_model_file_path)

    buildContentBaseModel(original_rdd=user_graph_input_lines,
                          export_model_file_path=export_content_model_file_path)

    print("Duration: %d s." % (time.time() - start))
