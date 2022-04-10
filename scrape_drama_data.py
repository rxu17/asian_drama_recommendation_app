''' Name: scrape_drama_data.py
    Description: Scrape drama data and metadata from mydramalist.com
        for further data processing, text cleaning and later 
        analysis and machine learning
    Arguments:
        is_scrape_parallel(bool) : whether to run scrapping code in parallel or not, 
                Allowed Values: ["True", "False"]
        save_path(str) : filepath to save scrapped data
        
    How to Use: python scrape_drama_data.py <is_scrape_parallel> <save_path>
    Example:
                python scrape_drama_data.py True /Users/some_user/Documents/
    Contributers: rxu17
'''
import sys
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import logging
import multiprocessing

# set logging format
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

MAIN_URL = "https://mydramalist.com/"


def verify_url(url : str) -> bool:
    """Whether url is valid or not
    Args:
        url (str): provided url
    Returns:
        bool: True, if url is valid else False
    """
    response = requests.get(url)
    if response.status_code == 200:
        return(True)
    else:
        return(False)
    
    
def get_all_page_urls() -> list:
    """Get list of all pages to scrape through

    Returns:
        list: integers of all page numbers
    """
    # here type is 68 which is dramas ONLY
    page_temp = "{}search?adv=titles&ty=68".format(MAIN_URL)
    # cutoff for ratings is at 5.0 to save space
    all_pos_ratings = list(np.round(np.arange(5, 10.1, 0.1),1))
    page_temps = [
        "{}&rt={},{}".format(page_temp, rating, rating) 
        for rating in all_pos_ratings
        ]
    
    all_page_urls = []
    for page_url in page_temps:
        page = requests.get(page_url)
        soup = BeautifulSoup(page.content, "html.parser")
        try:
            page_nums = soup.findAll('li', attrs = {'class':'page-item last'})[0].find("a").get("href")
            page_nums = list(range(1, (int(page_nums.split("page=")[-1]))+1))
        except:
            try:
                # if max pages button doesn't exist, try 2nd to last button
                max_pages = soup.find("ul", attrs={"class": "pagination"}).findAll("a", attrs={"class": "page-link"})
                max_pages = int(max_pages[len(max_pages)-2].text)
                page_nums = list(range(1, max_pages+1))
            except:
                # assume if we can't find pages to navigate, it only has one page
                page_nums = [1]
        page.close()
        all_page_urls += ["{}&page={}".format(page_url, pg) for pg in page_nums]
        logging.info("Got page numbers!")
    return(all_page_urls)


def get_page_nums() -> list:
    """Get list of all pages to scrape through

    Returns:
        list: integers of all page numbers
    """
    page = requests.get("{}/shows".format(MAIN_URL))
    soup = BeautifulSoup(page.content, "html.parser")
    page_nums = soup.findAll('li', attrs = {'class':'page-item last'})[0].find("a").get("href")
    page_nums = list(range(1, int("".join([i for i in page_nums if i.isdigit()]))+1))
    logging.info("Got page numbers!")
    page.close()
    return(list(range(1,501)))
    #return(list(set(page_nums)))


def get_drama_info(page_arg : object = None) -> pd.DataFrame:
    """Looks at list of all shows, movies, actors, pages, scrapes all available info
       and ONLY proceeds to scrape metadata IF the object is a DRAMA

    Args:
        page_arg (object): page number to scrape info from

    Returns:
        pd.DataFrame: table for page of all dramas
    """
    if isinstance(page_arg, int):
        page_url = "{}/shows/?page={}".format(MAIN_URL, page_arg)
    else:
        page_url = page_arg
    drama_urls = {'title':[], 'show_type':[], 'url':[]}
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")
    cell_content = soup.findAll('div', attrs={'class':"col-xs-9 row-cell content"})
    for cell in cell_content:
        # try to get show type, otherwise log info, likely actor
        try:
            show_type = cell.find('span', attrs = {'class':'text-muted'}).text
        except:
            logging.info("Can't find show type for {}".format(
                (cell.find('h6', attrs = {'class':'text-primary title'}).text)))
            continue
        # only add dramas to our list
        if 'Drama' in show_type:
            title = cell.find('h6', attrs = {'class':'text-primary title'})
            drama_urls['title'] = title.text
            drama_urls['show_type'] = show_type
            
            # verify url
            url = "{}{}".format(MAIN_URL, title.find("a").get("href"))
            if not verify_url(url):
                drama_urls['url'] = "needs_cleaning:{}".format(url)
            else:
                drama_urls['url'] = url
                drama_urls.update(get_drama_metadata(url))
        else:
            continue
    logging.info("Got drama url!")
    logging.info(page_arg)
    page.close()
    return(pd.DataFrame(drama_urls))


def get_drama_metadata(url : str) -> dict:
    """Scrape metadata such as watchers, rating, tas, synopsis
    from individual drama pages given by url

    Args:
        url (str): url of individual drama page to scrape from
    
    Returns:
        dict: dictionary of scrapped metadata data
    """
    drama_page = requests.get(url)
    soup = BeautifulSoup(drama_page.content, "html.parser")
    page_det = soup.find("div", attrs={"id":"show-detailsxx"})
    # get ratings
    rating = soup.find("div", attrs = {'class':"col-film-rating"}).text
    # get number of watchers
    try:
        watchers = page_det.findAll("div", attrs = {"class":"hfs"})[1].find("b").text
    except:
        watchers = np.nan
    # get synopsis
    synopsis = soup.find("div", attrs = {"class":"show-synopsis"}).find("span").text
    # get tags
    tags = soup.find("li", attrs = {"class":"list-item p-a-0 show-tags"}).findAll("a")
    tags = [tag.text for tag in tags]
    # get genres
    try:
        gens = soup.find("li", attrs = {"class":"list-item p-a-0 show-genres"}).findAll("a")
        gens = [gen.text for gen in gens]
    except:
        gens = []
    # get num of raters, ranking and popularity
    stat_box = soup.findAll("div", attrs = {"class":"box clear hidden-sm-down"})[1]
    stat_list =  stat_box.findAll("li", attrs = {"class":"list-item p-a-0"})
    raters = stat_list[0].find("span", attrs = {"class":"hft"}).text
    rank = stat_list[1].text
    pop = stat_list[2].text
    
    drama_metadata= {'rating':[rating], 'watchers':[watchers], 'synopsis':[synopsis],
                     'tags':[tags],'genres':[gens], 'raters':[raters],
                     'rank':[rank], 'popularity':[pop]}
    drama_page.close()
    # TODO: scrape reviews
    return(drama_metadata)


def scrape_serially() -> pd.DataFrame:
    """Scrapes data from MyDramaList one by one, serially

    Returns:
        pd.DataFrame: table of all scrapped urls and metadata
    """
    #page_nums = get_page_nums()
    page_urls = get_all_page_urls()
    drama_scrapped = pd.DataFrame({'title':[], 'show_type':[], 'url':[]})
    for page in page_urls:
        drama_scrapped = pd.concat([drama_scrapped, get_drama_info(page_arg = page)])
    return(drama_scrapped)
    
    
def scrape_in_parallel() -> pd.DataFrame:
    """Scrapes data from MyDramaList in parallel
        Default is 5 parallel processes

    Returns:
        pd.DataFrame: table of all scrapped urls and metadata
    """
    #args = get_page_nums()
    args = get_all_page_urls()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        results = pool.map(get_drama_info, args)
    drama_scrapped = pd.concat(results)
    return(drama_scrapped)


if __name__ == '__main__':
    is_scrape_parallel = True #sys.argv[0]
    #save_path = sys.argv[1]
    save_path = "/Users/rexxx"
    drama_scrapped = scrape_in_parallel() if is_scrape_parallel else scrape_serially()
    drama_scrapped.to_csv("{}/MDL_scrapped_data.csv".format(save_path))
    logging.info("Scrapped data saved to {}".format(
        "{}/MDL_scrapped_data.csv".format(save_path)))
    