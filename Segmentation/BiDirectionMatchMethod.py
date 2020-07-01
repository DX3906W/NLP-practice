from Segmentation.MaximumMatchMethod import MM
from Segmentation.ReverseMaximumMatchMethod import RMM


class BiMM(object):

    def __init__(self):

        self.window_size = 6
        self.word_list = ['研究', '研究生', '生命', '命', '的', '起源']

    def cut(self, sentence):
        tokenizer_mm = MM()
        tokenizer_rmm = RMM()

        result_mm = tokenizer_mm.cut(sentence)
        result_rmm = tokenizer_rmm.cut(sentence)

        if result_mm == result_rmm:
            print('true')
        else:
            if len(result_mm) > len(result_rmm):
                return result_rmm
            elif len(result_mm) == len(result_rmm):
                if self.count_char(result_mm) > self.count_char(result_rmm):
                    return result_rmm
                else:
                    return result_mm
            else:
                return result_mm

    def count_char(self, segmentation):
        count = 0
        for word in segmentation:
            if len(word) == 1:
                count = count + 1

        return count


if __name__ == '__main__':
    tokenizer = BiMM()
    result = tokenizer.cut('研究生命的起源')

    print(result)
