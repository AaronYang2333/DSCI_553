__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '4/29/2020 11:21 PM'

import json
import os
import sys
import time

from pyspark import SparkConf, SparkContext

USER_ID = 'user_id'
USER_ID_1 = 'u1'
USER_ID_2 = 'u2'
BUSINESS_ID = 'business_id'
SIMILARITY = 'sim'
SCORE = 'stars'
TARGET = "target"
IDOL = "idol"
AVG_BUSINESS_STAR = 3.7961611526341503
AVG_USER_STAR = 3.7961611526341503

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


def export2File(json_array, file_path):
    """
    export json content to a file
    :param json_array: a list of dict
    :param file_path: output file path
    :return: nothing, but a file
    """
    with open(file_path, 'w+') as output_file:
        for item in json_array:
            output_file.writelines(json.dumps(item) + "\n")


def makePrediction(mixed, data_dict, avg_score_dict=None,
                   reversed_index_dict=None):
    """

    :param mixed: tuple(uidx, [tuple(uidx, score), ...])  user_based
    :param data_dict:
    :param avg_score_dict:
    :param reversed_index_dict:
    :return: tuple(bidx, [(score, similarity),(),()])
    """
    target_uid = mixed[0]  # uidx
    target_uid_str = reversed_index_dict.get(target_uid, "UNK")
    mixed_uids_score_list = list(mixed[1])  # list of tuple(uidx, score)
    result = list()
    for uids_score in mixed_uids_score_list:
        if target_uid < uids_score[0]:
            key = tuple((target_uid, uids_score[0]))
        else:
            key = tuple((uids_score[0], target_uid))

        other_uid_str = reversed_index_dict.get(uids_score[0], "UNK")
        avg_score = avg_score_dict.get(other_uid_str, AVG_BUSINESS_STAR)
        # score, avg_score, similarity between users
        result.append(tuple((uids_score[1], avg_score, data_dict.get(key, 0))))

    numerator = sum(map(lambda item: (item[0] - item[1]) * item[2], result))
    if numerator == 0:
        return tuple((target_uid, avg_score_dict.get(target_uid_str, AVG_BUSINESS_STAR)))
    denominator = sum(map(lambda item: abs(item[2]), result))
    if denominator == 0:
        return tuple((target_uid, avg_score_dict.get(target_uid_str, AVG_BUSINESS_STAR)))

    return tuple((target_uid,
                  avg_score_dict.get(target_uid_str, AVG_USER_STAR) + (numerator / denominator)))


def makeGuess(bid_uid_pair):
    """
    predict user's rating score based on theirs friends
    :param bid_uid_pair: tuple(bid, uid)
    :return: tuple((bid, uid), score)
    """
    target_bid_str = bid_uid_pair[0]  # bidx
    target_uid_str = bid_uid_pair[1]  # uidx
    target_bid = bus_index_dict.get(target_bid_str, -1)
    if target_bid == -1:
        user_avg = user_avg_dict.get(target_uid_str, AVG_USER_STAR)
        predict_score = 0.1 * AVG_BUSINESS_STAR + 0.9 * user_avg
        return tuple((bid_uid_pair, round(predict_score)))

    # {('aBfT6rp0tfhEyc7fWeuYJA':[43473, 2218, 71746, 70143, 35056])
    friend_list = friend_circle_dict.get(target_uid_str, list())
    if len(friend_list) > 0:
        # if this user has some friends
        accumulator = 0
        for friend in friend_list:
            accumulator += train_data_dict.get((friend, target_bid),
                                               user_avg_dict.get(reversed_index_user_dict[friend], AVG_USER_STAR))
        friend_avg = float(accumulator / len(friend_list))
        business_score = bus_avg_dict.get(target_bid_str, AVG_BUSINESS_STAR)
        predict_score = 0.75 * friend_avg + 0.25 * business_score
        return tuple((bid_uid_pair, round(predict_score)))
    else:
        # if this user doesn't have any friends
        business_score = bus_avg_dict.get(target_bid_str, AVG_BUSINESS_STAR)
        predict_score = 0.2 * AVG_USER_STAR + 0.8 * business_score
        return tuple((bid_uid_pair, round(predict_score)))


def computeAvgScore(target_bid, target_uid):
    """
    compute avg score
    :param target_bid:  uid => (,121)
    :param target_uid:  uid => (12313,)
    :return:
    """

    target_bid_str = reversed_index_bus_dict.get(target_bid, "UNK")
    target_uid_str = reversed_index_user_dict.get(target_uid, "UNK")

    return (0.62 * bus_avg_dict.get(target_bid_str, AVG_BUSINESS_STAR)
            + 0.38 * user_avg_dict.get(target_uid_str, AVG_USER_STAR))


def switching(score1, score2):
    """
    switch different model when it comes to lower score
    :param score1:
    :param score2:
    :return:
    """
    if score2 > 1.5:
        if score2 > 4.5:
            return round(score2)
        return score2
    else:
        return round(0.8 * score1 + 0.2 * score2)


if __name__ == '__main__':
    start = time.time()
    # test_file_path = "../resource/asnlib/publicdata/test_review.json"
    # output_file_path = "../out/result2.predict"

    train_file_path = "../resource/asnlib/publicdata/train_review.json"
    export_cf_model_file_path = "./user-based.model"
    export_content_model_file_path = "./content-based.model"
    export_user_avg_model_file_path = "./user-avg.model"
    export_bus_avg_model_file_path = "./bus-avg.model"

    test_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    conf = SparkConf().setMaster("local[*]") \
        .setAppName("ay_inf_553_project_predict") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    # ========================== Build ID-Index Dict ===============================
    train_input_lines = sc.textFile(train_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: (kv[USER_ID], kv[BUSINESS_ID], kv[SCORE])).persist()

    user_index_dict = train_input_lines.map(lambda kvv: kvv[0]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_user_dict = {v: k for k, v in user_index_dict.items()}

    bus_index_dict = train_input_lines.map(lambda kvv: kvv[1]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    reversed_index_bus_dict = {v: k for k, v in bus_index_dict.items()}

    # ========================== Read AVG Info Dict ===============================
    # read avg info from json file and convert it into dict
    # dict(uid_str: avg_score)
    # => {"MHiKdBFx4McRQONnuMbByw": 3.857142857142857, ...}
    user_avg_dict = sc.textFile(export_user_avg_model_file_path) \
        .map(lambda row: json.loads(row)).map(lambda kv: dict(kv)) \
        .flatMap(lambda kv_items: kv_items.items()) \
        .collectAsMap()

    # read avg info from json file and convert it into dict
    # dict(bid_str: avg_score)
    # => {"AtD6B83S4Mbmq0t7iDnUVA": 4.393401015228426, ...}
    bus_avg_dict = sc.textFile(export_bus_avg_model_file_path) \
        .map(lambda row: json.loads(row)).map(lambda kv: dict(kv)) \
        .flatMap(lambda kv_items: kv_items.items()) \
        .collectAsMap()

    # ========================== Read CF Model ===============================
    # user based cf model
    # dict((uidx_pair): similarity)
    # => {(7415, 7567): 0.11736313170325506, (7415, 9653): 0.5222329678670935 ...}
    uid_pair_sim_dict = sc.textFile(export_cf_model_file_path) \
        .map(lambda row: json.loads(row)) \
        .map(lambda kvv: {(user_index_dict[kvv[USER_ID_1]],
                           user_index_dict[kvv[USER_ID_2]]): kvv[SIMILARITY]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    # ========================== Read Content Base Model =========================
    # content based model
    # dict((uid_str): idol_idx)
    # => {('aBfT6rp0tfhEyc7fWeuYJA': [43473, 2218, 71746, 70143, 35056])
    friend_circle_dict = sc.textFile(export_content_model_file_path) \
        .map(lambda row: json.loads(row)) \
        .map(lambda kvv: {kvv[TARGET]: [user_index_dict.get(idol, -1)
                                        for idol in list(kvv[IDOL])]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    # ========================== Read Training Data ===============================
    # dict((uidx, bidx): score)])
    # => {(6372, 5807): 5.0), ((6372,13378): 1.0), ((6372,14411): 5.0)...}
    train_data_dict = train_input_lines \
        .map(lambda kvv: {(user_index_dict[kvv[0]], bus_index_dict[kvv[1]]): kvv[2]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    # find the set of user who rated item i before
    # tuple(bidx, [tuple(uidx, score)])
    # => [(6372, [(5807, 5.0), (13378, 1.0), (14411, 5.0)...]), ()]
    visited_bus_rdd = train_input_lines \
        .map(lambda kvv: (bus_index_dict[kvv[1]], (user_index_dict[kvv[0]], kvv[2]))) \
        .groupByKey() \
        .map(lambda bid_uidxs: (bid_uidxs[0], [(uid_score[0], uid_score[1])
                                               for uid_score in list(set(bid_uidxs[1]))]))

    # ========================== Make Prediction ===============================
    # read test file and tokenized uid and bidx from test file
    # tuple(bidx, uidx)
    # => [(4871, 24954), (7557, 17243), (4593, 16426), (6791, 23814), (4383, 377)]
    test_data_rdd = sc.textFile(test_file_path).map(lambda row: json.loads(row)) \
        .map(lambda kv: (bus_index_dict.get(kv[BUSINESS_ID], kv[BUSINESS_ID]),
                         user_index_dict.get(kv[USER_ID], kv[USER_ID])))

    # normal user_business_pair
    # both user idx and business idx exist in training data
    normal_pair_rdd = test_data_rdd \
        .filter(lambda bid_uid: isinstance(bid_uid[0], int) and isinstance(bid_uid[1], int)) \
        .leftOuterJoin(visited_bus_rdd) \
        .mapValues(lambda mixed: makePrediction(mixed=tuple(mixed),
                                                data_dict=uid_pair_sim_dict,
                                                avg_score_dict=user_avg_dict,
                                                reversed_index_dict=reversed_index_user_dict)) \
        .map(lambda kvv: ((kvv[1][0], kvv[0]), (kvv[1][1], computeAvgScore(kvv[0], kvv[1][0])))) \
        .mapValues(lambda score_avg: switching(score_avg[0], score_avg[1])) \
        .map(lambda bid_uid_score: {"user_id": reversed_index_user_dict[bid_uid_score[0][0]],
                                    "business_id": reversed_index_bus_dict[bid_uid_score[0][1]],
                                    "stars": bid_uid_score[1]})

    # cold_start user_business_pair there are three scenarios:
    # tuple(bid, uid) ('AT_xDv2Lm5K7VtER1fp-SA', 'BSc3ubSD_URmTYJw0BIiTA')
    # 1. (6069, 'BSc3ubSD_URmTYJw0BIiTA') unknown user => we find their famous friends to help to make prediction
    # 2. ('HrOmA5pHWxLjE5T59WOv1g', 65871) unknown business => we just use avg business
    # 3. ('xx', 'xxx') or even new user went to the new business
    cold_start_pair_rdd = test_data_rdd \
        .filter(lambda bid_uid: isinstance(bid_uid[0], str) or isinstance(bid_uid[1], str)) \
        .map(lambda bid_uid: (bid_uid[0] if isinstance(bid_uid[0], str)
                              else reversed_index_bus_dict[bid_uid[0]],
                              bid_uid[1] if isinstance(bid_uid[1], str)
                              else reversed_index_user_dict[bid_uid[1]])) \
        .map(lambda bid_uid: makeGuess(bid_uid)) \
        .map(lambda bid_uid_score: {"user_id": bid_uid_score[0][1],
                                    "business_id": bid_uid_score[0][0],
                                    "stars": bid_uid_score[1]})

    output_pair = normal_pair_rdd.union(cold_start_pair_rdd)
    export2File(output_pair.collect(), output_file_path)
    print("Duration: %d s." % (time.time() - start))
