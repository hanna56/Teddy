from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd
import requests

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options)

total_page = 5  #  한페이지당 포스트 7개
keyword_list=["공포 소설", "두려움 소설"]

for keyword in keyword_list:
    url_list = []
    for i in range(total_page):
        i=i+1
        keyword_url="https://section.blog.naver.com/Search/Post.nhn?pageNo="+str(i)+"&rangeType=ALL&orderBy=sim&keyword="+keyword
        driver.get(keyword_url)
        time.sleep(0.4)
        
        # URL 크롤링 시작
        titles = "a.sh_blog_title._sp_each_url._sp_each_title"
        titles="#content > section > div.area_list_search > div:nth-child(1) > div > div.info_post > div.desc > a.text"

        url_raws = driver.find_elements_by_class_name("desc_inner")
        
        for url_raw in url_raws:
            url = url_raw.get_attribute('ng-href')   
            url_list.append(url)

    
    # print(len(url_list))
    df = pd.DataFrame({'url':url_list})

    # 저장하기
    df.to_excel(keyword+" url.xlsx", index=False)

    # 블로그 텍스트 크롤링
    blog_text_list=[]

    for url in url_list:
        blogId=url[23:-13]
        blogNo=url[-12:]

        real_blog_post_url="https://blog.naver.com/PostView.nhn?blogId="+blogId+"&logNo="+blogNo+"&from=search&redirect=Log&widgetTypeCall=true&directAccess=false"
        try:
            res = requests.get(real_blog_post_url)
            html = BeautifulSoup(res.content, 'html.parser')
            text = html.find('div', {'class':'se-main-container'}).get_text()
            blog_text_list.append(text)
        except:
            pass

    # print(len(blog_text_list))
    df = pd.DataFrame({'text':blog_text_list})

    # 저장하기
    df.to_excel(keyword+"_new text.xlsx", index=False)

driver.quit()