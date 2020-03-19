from xml.etree import ElementTree
import urllib
from urllib import request
import pandas as pd
import numpy as np
import json


class GetData:
    def __init__(self):
        self.Borlange = {}
        self.Mora = {}
        self.Karlstad = {}
        self.Gavle = {}
        self.Vasteras = {}

        self.response_Borlange1 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.466671,15.41667&date=2019-06-01&enddate=2019-06-30&key=ba2c0154db0444a5b2d222033200601')
        self.response_Borlange2 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.466671,15.41667&date=2019-07-01&enddate=2019-07-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Borlange3 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.466671,15.41667&date=2019-08-01&enddate=2019-08-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Borlange4 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.466671,15.41667&date=2019-09-01&enddate=2019-09-30&key=ba2c0154db0444a5b2d222033200601')
        self.getInfo(self.response_Borlange1, self.Borlange)
        self.getInfo(self.response_Borlange2, self.Borlange)
        self.getInfo(self.response_Borlange3, self.Borlange)
        self.getInfo(self.response_Borlange4, self.Borlange)

        self.response_Mora1 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=61.007038,14.54316&date=2019-06-01&enddate=2019-06-30&key=ba2c0154db0444a5b2d222033200601')
        self.response_Mora2 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=61.007038,14.54316&date=2019-07-01&enddate=2019-07-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Mora3 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=61.007038,14.54316&date=2019-08-01&enddate=2019-08-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Mora4 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=61.007038,14.54316&date=2019-09-01&enddate=2019-09-30&key=ba2c0154db0444a5b2d222033200601')
        self.getInfo(self.response_Mora1, self.Mora)
        self.getInfo(self.response_Mora2, self.Mora)
        self.getInfo(self.response_Mora3, self.Mora)
        self.getInfo(self.response_Mora4, self.Mora)

        self.response_Karlstad1 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.379299,13.50357&date=2019-06-01&enddate=2019-06-30&key=ba2c0154db0444a5b2d222033200601')
        self.response_Karlstad2 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.379299,13.50357&date=2019-07-01&enddate=2019-07-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Karlstad3 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.379299,13.50357&date=2019-08-01&enddate=2019-08-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Karlstad4 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.379299,13.50357&date=2019-09-01&enddate=2019-09-30&key=ba2c0154db0444a5b2d222033200601')
        self.getInfo(self.response_Karlstad1, self.Karlstad)
        self.getInfo(self.response_Karlstad2, self.Karlstad)
        self.getInfo(self.response_Karlstad3, self.Karlstad)
        self.getInfo(self.response_Karlstad4, self.Karlstad)

        self.response_Gavle1 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.674519,17.141741&date=2019-06-01&enddate=2019-06-30&key=ba2c0154db0444a5b2d222033200601')
        self.response_Gavle2 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.674519,17.141741&date=2019-07-01&enddate=2019-07-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Gavle3 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.674519,17.141741&date=2019-08-01&enddate=2019-08-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Gavle4 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=60.674519,17.141741&date=2019-09-01&enddate=2019-09-30&key=ba2c0154db0444a5b2d222033200601')
        self.getInfo(self.response_Gavle1, self.Gavle)
        self.getInfo(self.response_Gavle2, self.Gavle)
        self.getInfo(self.response_Gavle3, self.Gavle)
        self.getInfo(self.response_Gavle4, self.Gavle)

        self.response_Vasteras1 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.616169,16.552759&date=2019-06-01&enddate=2019-06-30&key=ba2c0154db0444a5b2d222033200601')
        self.response_Vasteras2 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.616169,16.552759&date=2019-07-01&enddate=2019-07-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Vasteras3 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.616169,16.552759&date=2019-08-01&enddate=2019-08-31&key=ba2c0154db0444a5b2d222033200601')
        self.response_Vasteras4 = urllib.request.urlopen(
            r'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q=59.616169,16.552759&date=2019-09-01&enddate=2019-09-30&key=ba2c0154db0444a5b2d222033200601')
        self.getInfo(self.response_Vasteras1, self.Vasteras)
        self.getInfo(self.response_Vasteras2, self.Vasteras)
        self.getInfo(self.response_Vasteras3, self.Vasteras)
        self.getInfo(self.response_Vasteras4, self.Vasteras)

    def getInfo(self, response, location):
        page = response.read()
        page = page.decode('utf-8')
        tree = ElementTree.fromstring(page)
        for weatherElement in tree.findall("weather"):
            day_hour = []
            for hour in weatherElement.findall("hourly"):
                day_hour.append(int(hour.find("tempC").text))
            location[weatherElement.find("date").text] = day_hour

    def getBorlangeInfo(self):
        return self.Borlange

    def getMoraInfo(self):
        return self.Mora

    def getKarlstadInfo(self):
        return self.Karlstad

    def getGavleInfo(self):
        return self.Gavle

    def getVasterasInfo(self):
        return self.Vasteras

    def getLastedTemp(self):

        #url_Borlange = "http://api.openweathermap.org/data/2.5/weather?id=2720382&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Mora = "http://api.openweathermap.org/data/2.5/weather?id=2691400&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Karlstad = "http://api.openweathermap.org/data/2.5/weather?id=2701680&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Gavle = "http://api.openweathermap.org/data/2.5/weather?id=2712411&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Vasteras = "http://api.openweathermap.org/data/2.5/weather?id=2664454&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"

        url_list = [url_Mora, url_Karlstad, url_Gavle, url_Vasteras]
        current_temp = []
        for item in url_list:
            urltojson = urllib.request.urlopen(item)
            response = urltojson.read()
            decodedResponse = response.decode("utf-8")
            jsondata = json.loads(decodedResponse)
            celsius = round(float(jsondata["main"]["temp"]) - 237.15)
            current_temp.append(celsius)

        return current_temp

    def getLastTemp(self):

        # url_Borlange = "http://api.openweathermap.org/data/2.5/weather?id=2720382&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Mora = "http://api.openweathermap.org/data/2.5/weather?id=2691400&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Karlstad = "http://api.openweathermap.org/data/2.5/weather?id=2701680&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Gavle = "http://api.openweathermap.org/data/2.5/weather?id=2712411&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"
        url_Vasteras = "http://api.openweathermap.org/data/2.5/weather?id=2664454&APPID=da5c9b841c0fbb4a1a8285a89c739cdb"

        url_list = [url_Mora, url_Karlstad, url_Gavle, url_Vasteras]
        current_temp = []
        for item in url_list:
            urltojson = urllib.request.urlopen(item)
            response = urltojson.read()
            decodedResponse = response.decode("utf-8")
            jsondata = json.loads(decodedResponse)
            celsius = round(float(jsondata["main"]["temp"]) - 237.15)
            current_temp.append(celsius)

        return current_temp

if __name__=="__main__":
    data = GetData()

    Borlange_dict = data.getBorlangeInfo()
    Mora_dict = data.getMoraInfo()
    Karlstad_dict = data.getKarlstadInfo()
    Gavle_dict = data.getGavleInfo()
    Vasteras_dict = data.getVasterasInfo()

    titles = ["Mora", "Karlstad", "Gavle", "Vasteras", "Label"]
    Borlange_date = list(Borlange_dict.keys())

    def get_csv_data(keys):
        csv_data = []
        for index, key in enumerate(keys):
            for time in range(0, 8):
                row_data = []

                if index==1:
                    break
                else:
                    label = 0
                    past_index = keys[index-1]

                    row_data.append(Mora_dict[key][time] - Mora_dict[past_index][time])
                    row_data.append(Karlstad_dict[key][time] - Karlstad_dict[past_index][time])
                    row_data.append(Gavle_dict[key][time] - Gavle_dict[past_index][time])
                    row_data.append(Vasteras_dict[key][time] - Vasteras_dict[past_index][time])

                    cur_temp = Borlange_dict[key][time]
                    past_temp = Borlange_dict[past_index][time]
                    if cur_temp - past_temp >= 2:
                        label = 0
                    elif past_temp - cur_temp >= 2:
                        label = 1
                    else:
                        label = 2
                    row_data.append(label)

                csv_data.append(row_data)

        return csv_data

    pointOfTime = []
    for index, date in enumerate(Borlange_date):
        if index == 0:
            continue
        for i in range(0, 24, 3):
            pointOfTime.append(date + "-"  + str(i))

    data_0 = pd.DataFrame(columns=titles, index=pointOfTime, data=get_csv_data(Borlange_date))
    data_0.to_csv("F://assignment//data//dataSet.csv",  sep=',')

