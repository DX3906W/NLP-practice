
class RMM(object):

    def __init__(self):
        self.window_size = 6
        self.word_list = ['研究生', '研究', '生命',  '命', '的', '起源']

    def cut(self, sentence):
        result = []
        index = 0

        while index < len(sentence):
            for size in range(1, self.window_size+1):
                piece = sentence[index: index+size]
                if piece in self.word_list:
                    result.append(piece)
                    index = index + size

        return result


if __name__ == '__main__':
    tokenizer = RMM()

    segmentation = tokenizer.cut('研究生命的起源')
    print(segmentation)

