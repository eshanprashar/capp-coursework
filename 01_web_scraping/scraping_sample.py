import sys
import json
import lxml.html
from parks.utils import make_request, make_link_absolute


def scrape_park_page(url):
    """
    This function takes a URL to a park page and returns a
    dictionary with the title, address, description,
    and history of the park.

    Parameters:
        * url:  a URL to a park page

    Returns:
        A dictionary with the following keys:
            * url:          the URL of the park page
            * name:         the name of the park
            * address:      the address of the park
            * description:  the description of the park
            * history:      the history of the park
    """
    park_info = {}
    page_response = make_request(url)
    root = lxml.html.fromstring(page_response.text)
    park_name = root.xpath("//h2[@class='section']")[0].text_content()
    park_address = root.xpath("//p[@class='address']")[0].text_content()
    common_sibling = root.xpath("//h3[@class='block-title']//following-sibling::div[@class='block-text']")
    park_desc = common_sibling[0].text_content()
    if len(common_sibling) == 2:
        park_his = common_sibling[1].text_content()
    else:
        park_his = ""
    park_info.update({"url": url, "name" :park_name, "address": park_address, "description" : park_desc, "history" : park_his})
    return park_info
    
def get_park_urls(url):
    """
    This function takes a URL to a page of parks and returns a
    list of URLs to each park on that page.

    Parameters:
        * url:  a URL to a page of parks

    Returns:
        A list of URLs to each park on the page.
    """
    urls = []
    page_response = make_request(url)
    root = lxml.html.fromstring(page_response.text)
    #leveraging fact that desired href element is the only one within a table td
    td_url_lst = root.xpath("//td/a/@href")
    for href_element in td_url_lst:
        complete_url = make_link_absolute(href_element, url)
        urls.append(complete_url)
    return urls
    
def get_next_page_url(url):
    """
    This function takes a URL to a page of parks and returns a
    URL to the next page of parks if one exists.

    If no next page exists, this function returns None.
    """
    link = None
    page_response = make_request(url)
    root = lxml.html.fromstring(page_response.text)
    #div elements with href can only have Previous and/or Next buttons
    link_lst = root.xpath("//div/a/@href")
    # if both previous and next exist
    if len(link_lst) == 2:
        link = make_link_absolute(link_lst[1],url)
    # page 2 will not have a previous button
    elif link_lst[0] == "?page=2":
        link = make_link_absolute(link_lst[0],url)
    return link

def crawl(max_parks_to_crawl):
    """
    This function starts at the base URL for the parks site and
    crawls through each page of parks, scraping each park page
    and saving output to a file named "parks.json".

    Parameters:
        * max_parks_to_crawl:  the maximum number of pages to crawl
    """
    list_page_url = "https://scrapple.fly.dev/parks"
    parks = []
    parks_crawled = 0
    while parks_crawled < max_parks_to_crawl:
        park_urls = get_park_urls(list_page_url)
        for park in park_urls:
            parks.append(scrape_park_page(park))
            parks_crawled += 1
        list_page_url = get_next_page_url(list_page_url)
        if list_page_url is None:
            break
    print("Writing parks.json")
    with open("parks.json", "w") as f:
        json.dump(parks, f, indent=1)
    f.close()


if __name__ == "__main__":
    """
    Tip: It can be convenient to add small entrypoints to submodules
         for ease of testing.

    In this file, we call scrape_park_page with a given URL and pretty-print
    the output.

    This allows testing that function from the command line with:

    $ python -m parks.crawler https://scrapple.fly.dev/parks/4

    Feel free to modify/change this if you wish, you won't be graded on this code.
    """
    from pprint import pprint

    if len(sys.argv) != 2:
        print("Usage: python -m parks.crawler <url>")
        sys.exit(1)
    result = scrape_park_page(sys.argv[1])
    pprint(result)