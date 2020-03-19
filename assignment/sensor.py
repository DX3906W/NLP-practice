import os
import glob
import time
import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.utils import COMMASPACE
import RPi.GPIO as GPIO
from datetime import datetime
from modeling import KNN_Model
import numpy as np
from dataHouse import GetData

class Sensor:
    def __init__(self):

        self.os.system("modprobe w1_gpio")
        self.os.system("modprobe w1_therm")

        self.devicelist = glob.glob("/sys/bus/w1/devices/28*")
        self.devicefile = self.devicelist[0]+"/w1_slave"

        self.dayTemp = []

        self.SENDER = 'junweigong0571@gmail.com'
        self.SMTP_SERVER = 'smtp.gmail.com'
        self.USER_ACCOUNT = {'username': 'junweigong0571@gmail.com', 'password': 'qazokmGJWlove'}
        self.receivers = ['gjw0571@qq.com']

        self.dataProcessor = GetData()
        self.model = KNN_Model()

        GPIO.cleanup()

        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(3, GPIO.OUT)
        GPIO.output(3, GPIO.LOW)

        GPIO.setup(5, GPIO.OUT)
        GPIO.output(5, GPIO.LOW)

        GPIO.setup(7, GPIO.OUT)
        GPIO.output(7, GPIO.LOW)

    def sendEmail(self, currentDate, change):
        msg_root = MIMEMultipart()
        msg_root['Subject'] = "Sensor alert(no-reply)"
        msg_root['To'] = COMMASPACE.join(self.receivers)
        text = "Today is {}, the temperature will {} in the next three hours.".format(currentDate, change)
        msg_text = MIMEText(text, 'html', 'utf-8')
        msg_root.attach(msg_text)

        smtp = smtplib.SMTP('smtp.gmail.com:587')
        smtp.ehlo()
        smtp.starttls()
        smtp.login(self.USER_ACCOUNT['username'], self.USER_ACCOUNT['password'])
        smtp.sendmail(self.SENDER, self.receivers, msg_root.as_string())

    def run(self):
        while True:
            fileobj = open(self.devicefile, 'r')

            lines = fileobj.readlines()
            fileobj.close()

            tempdata = lines[1].split("=")

            sensorTemp = float(tempdata[1])

            sensorTemp = sensorTemp / 1000

            self.dayTemp.append(sensorTemp)

            if len(self.dayTemp) == 6:
                lowestTemp = min(self.dayTemp)
                highestTemp = max(self.dayTemp)
                averageTemp = int(sum(self.dayTemp) / 6)
                del self.dayTemp[:]

                now_date_time = datetime.datetime.now()
                date = now_date_time.strftime("%Y-%m-%d")

                url = "http://130.243.35.86/lab5/receiveTemp.php?date=" + date + "&highestTemp=" + str(
                    highestTemp) + "&lowestTemp=" + str(
                    lowestTemp) + "&averageTemp=" + str(averageTemp)
                requests.get(url)

                lastUrl = r"http://130.243.35.86/lab5/receiveTemp.php";
                r = requests.get(lastUrl)

                test_data = np.array(self.dataProcessor.getLastedTemp()) - np.array(self.dataProcessor.getLastTemp())
                prediction = np.argmax(self.model.predict(test_data.ravel()))
                self.setLight(prediction)

            time.sleep(1800)

    def labling(self, cur, past):
        if cur - past >= 2:
            return 0
        elif past - cur >= 2:
            return 1
        else:
            return 2

    def setLight(self, key):

        #three cases for traffic lights
        date = datetime.now()

        if key == 2:
            GPIO.output(3, GPIO.LOW)
            GPIO.output(5, GPIO.LOW)
            GPIO.output(7, GPIO.HIGH)
        if key == 1:
            GPIO.output(3, GPIO.LOW)
            GPIO.output(7, GPIO.LOW)
            GPIO.output(5, GPIO.HIGH)
            change = "fall"
            self.sendEmail(date, change)

        if key == 0:
            GPIO.output(5, GPIO.LOW)
            GPIO.output(7, GPIO.LOW)
            GPIO.output(3, GPIO.HIGH)
            change = "rise"
            self.sendEmail(date, change)

