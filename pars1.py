
#!/usr/bin/python3
from bs4 import BeautifulSoup
import requests as req
import logging
import re
import pprint
from ru_soundex.soundex import RussianSoundex
soundex = RussianSoundex(delete_first_letter=True)
resp = req.get("http://www.shhyolkovo.ru/news/")
soup = BeautifulSoup(resp.text, 'lxml')
news_table = soup.find("div", {'class': "block-12-news_wrap"})
items = news_table.find_all("li", {'class':"block-12-news-one"})
Ideal_table = [ "", "Работа", "Благотворительность", "Соцзащита", "Правопорядок", "Молодёжь", "ЖКХ", "Самоуправление", "Мои документы", "Безопасность", "Строительство", "Медицина", "Экология", "Транспорт", "Наука и техника", "Общество", "Экономика", "Спорт", "Туризм и отдых", "Культура"]
for item in items:
	news_pt = item.find("div", {'class':"overflow"})
	news_link = item.find("div", {'class': "overflow"}).find('a').get('href')
	news_theme = news_pt.find("div", {'class':"block-12-news-one-img"}).find('span').text.strip()
	news_name = item.find("div", {'class': "overflow"}).find('h4').text
	resp1 = req.get("http://www.shhyolkovo.ru" + news_link)
	soup1 = BeautifulSoup( resp1.text, 'lxml')
	list = soup1.find("div", {'class': "dop_inner_content_withoutMenu"}).find_all('p')
	fultxt=""
	for i in list:
		fultxt = fultxt + i.text
	fultxt = fultxt.replace("."," ")
	fultxt = fultxt.replace(","," ")
	fultxt = fultxt.replace("«"," ")
	fultxt = fultxt.replace("»"," ")
	fultxt = fultxt.replace("-"," ")
	fultxt = fultxt.replace("("," ")
	fultxt = fultxt.replace(")"," ")
	fultxt = fultxt.replace("?"," ")
	fultxt = fultxt.replace("©"," ")
	fultxt = fultxt.replace(":"," ")
	fultxt = fultxt.replace("1", " ")
	fultxt = fultxt.replace("2", " ")
	fultxt = fultxt.replace("3", " ")
	fultxt = fultxt.replace("4", " ")
	fultxt = fultxt.replace("5", " ")
	fultxt = fultxt.replace("6", " ")
	fultxt = fultxt.replace("7", " ")
	fultxt = fultxt.replace("8", " ")
	fultxt = fultxt.replace("9", " ")
	fultxt = fultxt.replace("0", " ")
	fultxt = fultxt.replace("!"," ")
	fultxt = fultxt.split()
	A = [0]*500
	B = [0]*500
	theme = Ideal_table.index(news_theme)
	D = [0]*20
	D[0] = news_name
	D[theme] = 1
	Ideal_table.append(D)
	i=0
	for word in fultxt:
		if (len(word) > 3):
			sw = soundex.transform(word)[0:4]
			if ( sw in A):
				B[A.index(sw)]+=1
			else:
				A[i] = sw
				B[i] = 1
				i+=1
pprint.pprint(Ideal_table)

