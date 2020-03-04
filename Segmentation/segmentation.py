
from functools import reduce


def build_graph(word_list, sentence):

    graph = {}

    for index in range(len(sentence)):
        temp_list = []
        i = index

        while i <= len(sentence):
            temp = sentence[index: i]
            if temp in word_list:
                temp_list.append(i)
            i += 1
        graph[index] = temp_list
    return graph


def viterbi(word_prob, segments, sentence):
    record = [0 for i in range(len(segments))]
    routes = {}

    for edge in segments[0]:
        record[edge-1] = word_prob[sentence[0: edge]]
        routes[edge-1] = [0, edge]

    for i in range(1, len(segments)):
        temp_list = segments[i]

        for edge in temp_list:
            temp = record[i-1] * word_prob[sentence[i: edge]]
            if temp >  record[edge-1]:
                record[edge-1] = temp
                routes[edge-1] = routes[i-1][:]
                routes[edge-1].append(edge)

    route = routes[len(sentence)-1]
    return [sentence[route[i-1]: route[i]] for i in range(1, len(route))]


def word_segment_naive(word_prob, input_str):

    # TODO： 第一步： 计算所有可能的分词结果，要保证每个分完的词存在于词典里，这个结果有可能会非常多。
                    # 存储所有分词的结果。如果次字符串不可能被完全切分，则返回空列表(list)
                   # 格式为：segments = [["今天"，“天气”，“好”],["今天"，“天“，”气”，“好”],["今“，”天"，“天气”，“好”],...]
    segments = build_graph(word_prob, input_str)
    # TODO: 第二步：循环所有的分词结果，并计算出概率最高的分词结果，并返回
    best_segment = viterbi(word_prob, segments, input_str)

    best_score = reduce(lambda x,y:x * y, [word_prob[key] for key in best_segment])

    return best_segment


word_prob = {"北京":0.03,"的":0.08,"天":0.005,"气":0.005,"天气":0.06,"真":0.04,"好":0.05,"真好":0.04,"啊":0.01,"真好啊":0.02,
         "今":0.01,"今天":0.07,"课程":0.06,"内容":0.06,"有":0.05,"很":0.03,"很有":0.04,"意思":0.06,"有意思":0.005,"课":0.01,
         "程":0.005,"经常":0.08,"意见":0.08,"意":0.01,"见":0.005,"有意见":0.02,"分歧":0.04,"分":0.02, "歧":0.005}

segments = word_segment_naive(word_prob, "今天的课程内容很有意思")
print(segments)

segment = word_segment_naive(word_prob, "北京的天气真好啊")
print(segment)

segment = word_segment_naive(word_prob, "经常有意见分歧")
print(segment)
