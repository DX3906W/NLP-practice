
import pdfplumber
import numpy as np
import appConfig
import os

def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        print('='*10, 'Start to read pdf file, totally ', len(pdf.pages), ' pages', '='*10)
        pdf_text = []
        page_num = len(pdf.pages)
        for index, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            print('-'*10, 'reading ', index+1, ' page', '-'*10)
            pdf_text.append(page_text)

    return pdf_text

def write_txt(file_path, file_text):
    with open(file_path,  'w') as f:
        for page_text in file_text:
            f.write(page_text)

def get_data():
    print(appConfig.file_path)
    get_dir = os.listdir(appConfig.file_path)
    data = []
    print('='*10, 'totally ', len(get_dir), ' file(s)', '='*10)
    for index, file in enumerate(get_dir):
        txt_data = []
        sub_dir = os.path.join(appConfig.file_path, file)
        #print(sub_dir)
        print('-'*10, 'reading ', index+1, ' now', '-'*10)
        with open(sub_dir, 'r') as f:
            for line in f.readlines():
                txt_data.append(line.replace('\n', ''))

        data.append(' '.join(txt_data))
        print(txt_data)
    return data


if __name__ == '__main__':
    pdf_text1 = read_pdf("C:/Users/dell1/Downloads/take_home_task(Gekko)/take_home_task(Gekko)/example1/example1.pdf")
    pdf_text2 = read_pdf("C:/Users/dell1/Downloads/take_home_task(Gekko)/take_home_task(Gekko)/example2/example2.pdf")

    write_txt('F:/mygit/NLP-practice/Competition/Interview/data/file1.txt', pdf_text1)
    write_txt('F:/mygit/NLP-practice/Competition/Interview/data/file2.txt', pdf_text2)
