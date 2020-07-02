
class HMM(object):

    def __init__(self):
        self.model_file = '.data/hmm_model.pkl'

        self.state_list = ['B', 'M', 'E', 'S']

        self.load_para = False

    def try_load_model(self, trained):
        if trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True

        else:
            #状态转移概率
            self.A_dic = {}
            #发射概率：状态 -> 词语的条件概率
            self.B_dic = {}
            #状态的初始概率
            self.Pi_dic = {}
            self.load_para = False

    def train(self, path):
        self.try_load_model(False)

        #统计状态出现次数
        Count_dic = {}

        def init_parameters():
            for state in self.state_list:
                self.A_dic = {s: 0.0 for s in self.state_list}
                self.Pi_dic = 0.0
                self.B_dic = {}

                Count_dic[state] = 0.0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M']*(len(text)-2) + ['E']

            return out_text

        init_parameters()
        line_num = -1

        chars = set()
        with open(path, encoding='utf-8') as f:
            for line in f:
                line_num += 1

                line = line.strip()
                if not line:
                    continue

                char_list = [i for i in line if i != ' ']
                chars |= set(char_list)

                word_list = line.split()

                line_state = []
                for word in word_list:
                    line_state.extend(makeLabel(word))

                assert len(char_list) == len(line_state)

                for k, v in enumerate(line_state):
                    Count_dic[v] += 1
                    if k == 0:
                        self.Pi_dic[v] += 1
                    else:
                        self.A_dic[line_state[k-1]][v] += 1
                        self.B_dic[v][char_list[k]] = \
                            self.B_dic[v].get(char_list[k], 0) + 1.0

        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / Count_dic[k] for k1, v1 in v.items()}
                        for k, v in self.A_dic.items()}
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()}
                        for k, v in self.B_dic.items()}

        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

        return self
    """
    @:param
    text: the sentence used to cut word
    states: including all states
    start_p: the start probability of each states
    trans_p: the transformation probability from one state to another state
    emit_p: the probability of words at special state 
    """
    def viterbi(self, text, states, start_p, trans_p, emit_p):

        # path graph
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            V.append({})
            newpath = {}
            # check if the word existed in the emit probability matrix
            neverSeen = text[t] not in emit_p['S'].keys() and \
                text[t] not in emit_p['M'].keys() and \
                text[t] not in emit_p['E'].keys() and \
                text[t] not in emit_p['B'].keys()

            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                (prob, state) = max(
                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0)
                     for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = newpath
            if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text(-1), 0):
                (prob, state) = max([(V[len(text) - 1][y], y) for y in ['E', 'M']])
            else:
                (prob, state) = max([V[len(text) - 1][y], y] for y in states)

            return (prob, path[state])

    def cut(self, text):
        import os
        if not self.try_load_model():
            self.try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        begin, next = 0, 0

        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i + 1]
                next = i + 1
            elif pos == 'S':
                yield  char
                next = i + 1

        if next < len(text):
            yield text[next:]


if __name__ == '__main__':
    hmm = HMM()
    hmm.train()

    res = hmm.cut('')

    print(str(list(res)))



