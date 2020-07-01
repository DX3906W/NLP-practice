class MM(object):

    def __init__(self):

        self.window_size = 6
        self.word_list = ['研究', '研究生', '生命', '命', '的', '起源']

    def cut(self, sentence):
        result = []
        index = 0

        while index < len(sentence):
            for size in range(self.window_size, 0, -1):
                piece = sentence[index: index + size]
                if piece in self.word_list:
                    result.append(piece)
                    index = index + size

        return result


if __name__ == '__main__':
    sentence = '研究生命的起源'
    tokenizer = MM()

    segmentation = tokenizer.cut(sentence)
    print(segmentation)
