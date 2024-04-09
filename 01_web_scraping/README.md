# Scraping Chicago Parks

Over the course of the next few assignments, you will build parts of an application that allows people to look up information on city parks in Chicago.

In this part of the assignment, you will be building a scraper that crawls a simulacrum of the Chicago Parks website to construct a search index. The assignment will give you experience building a real-world crawler that works with HTML documents as well as additional practice composing a Python application.

Instead of scraping the real Chicago Parks website, we'll be scraping: <https://scrapple.fly.dev/parks>

(This is a site built just for this project, and therefore there is no risk of it changing overnight or a runaway scraper accidentally hitting the site too hard and causing problems for people outside this class. The data & relevant HTML were themselves scraped from <https://www.chicagoparkdistrict.com/about-us/history-chicagos-parks>.)

## Getting Started

You are given several files to start:

`parks/utils.py` contains several utility functions you can use while writing your scraper:

* `make_request(url)` this function should be used in lieu of `requests.get` for this assignment. It contains code to limit how many requests you make at once, and ensure you are not accidentally fetching data from domains outside the scope of the assignment.  It returns a `requests.Response` object you can use as if you had called `requests.get` directly.
* `make_link_absolute(rel_url, current_url)` converts relative URLs and fragments that you might encounter (e.g. `<a href='?page=3'>`) to full URLs.  The function's docstring gives more examples.

*You should not change any functions within utils.py*

`parks/crawler.py` is where you will complete the first part of the assignment as described below.

`parks/clean.py` is where you will complete the second part of the assignment as described below.

`parks/__main__.py` provides an entrypoint that will call your code in `crawler.py` and `clean.py`.

`tests/test_crawler.py` includes `pytest` tests that can help you verify the proper functioning of your crawler.

`tests/test_clean.py` includes `pytest` tests that can help you verify the proper functioning of your cleaning code.

## Part 1: Writing the Scraper

We'll be scraping data on individual parks from <https://scrapple.fly.dev/parks>

If you take a look at this page, you'll see that it consists of a list of parks, and at the bottom also has a pagination section.

In order to scrape this data, you will need to write several functions:

* A function to scrape all of the park URLs on a given page, `get_parks_urls`.
* A function to scrape all of the data on a given park's detail page. `scrape_park_page`.
* A function to get the next page of parks if one exists. `get_next_page_url`.
* A function to put the pieces together,
We will be writing a helper function to extract information from a single page.

### Individual Park Pages

I'd recommend starting with the `scrape_park_page` function. Within that function you'll be scraping an individual park page.

Each park page has a URL like:
<https://scrapple.fly.dev/parks/3>

For each of these pages, you will want to do the following:

1) Extract the park's name, this can be found in the `<h2>` element:

```html
...
  <div class="page-title">
    <h2 class="section">Abbot (Robert) Park</h2>
  </div>
...
```

2) Extract the park's address.  You'll see data on the page that looks like this:

```html
<span class="location-header-value">
  <p class="address">
          49 E. 95th St. <br>
            Chicago, IL 60628<br>
  </p>
</span>
```

You can extract this text from the `<p class="address">` tag. You do not need to worry about reformatting this text for now, just extract the location as it is provided.

3) Finally, you'll need to extract the raw text from the Description & History blocks.

The Description block looks like this:

```html
<div class="view-content">
  <h2><span id="Description"></span></h2>
  <h2 class="block-title"> Description </h2>
  <div class="block-text"><p>Located in the Roseland Community Area, Abbott Park totals 25&nbsp;acres and features a multi-purpose room and game room. Outside, the park offers four baseball diamonds, basketball, track and tennis courts, swimming pool, and two sprinklers. Many of these spaces are available for rental including our multi-purpose room and game room.</p>

  <p>Park-goers can participate in seasonal sports,&nbsp;cheerleading,&nbsp;aerobics, senior and teen clubs. On the cultural side, Abbott Park offers dance, music and movement. After school programs are offered throughout the school year, and during the summer youth can attend the Park District’s popular six-week day camp.Specialty camps, including Sports Camp, are also offered in the summer.</p>

  <p>In addition to programs, Abbott Park hosts fun special events throughout the year for the entire family including&nbsp;holiday events.&nbsp;</p>
   </div>
</div>
```

In this case, you'll need to find the element `<h2 class="block-title"> Description </h2>` and then
get the text of the following (sibling) `div class="block-text"` element.

You will find that the "History" section is very similar, just with "History" instead of description.

(*Note: History is not present for every park, in that case you should just have your data contain an empty string `""`.*)

If you'd like to run that function, you can do so from the ipython REPL, or by running

```shell
python -m parks.crawler https://scrapple.fly.dev/parks/3
```

Which is set up to allow you to scrape individual pages (see the `__main__` block at the end of the file).

### Getting Park Pages From Index

Now that you can scrape a single park, you'll want to write the `get_park_urls` function that gathers a list of the park urls.

Look at <https://scrapple.fly.dev/parks>. You'll see that each link on the page resembles:

```html
    <td>
    <a href="/parks/1">Details</a>
    </td>
```

You need to extract the `href` here.  Note that it is not a complete URL, but just a fragment.
You will need to use `utils.make_link_absolute` to get a full URL.

### Pagination

Finally, you will need to find the link to go to the next page. If there is a next page you'll see a link at the bottom like:

```html
  <a class="button" style="justify-self: end;" href="?page=2" title="next page">Next &raquo;</a>
```

For this site when no such link is found, you will know you've reached the end.

### Putting it Together

Finally, you'll complete the `crawl` method in this file.

This method should use the other methods you wrote (and any helper methods you deem necessary) to build a list of park dictionaries.

Once you have compiled a list of all park data, write the data to a file named "parks.json" using `json.dump`.

To make your file human-readable, I'd recommend using the following to add some indentation to the JSON file:

```python
json.dump(parks, fd, indent=1)
```

This adds newlines and indentations to the JSON file instead of having it all on one line. This doesn't affect how Python handles the data, but does make much nicer to look at in a text editor.

Note: It can be useful to limit how much data is scraped while testing, and for that purpose you'll see there is a variable `max_pages_to_crawl`.  Keep track of how many times you've called scrape_park_page and if the number equals `max_pages_to_crawl` stop scraping.

## Part 2: Post-Processing

Once the data is scraped, we will perform any needed transformations on the data.

This work will be done in `cleanup.py`. For this file we do not provide the needed imports, so you will need to decide what modules are necessary.

As discussed in class, separating the steps this way allows working on your processing pipeline without needing to re-run your scrapers.

### Address Normalization

When we extracted the address data before, it is likely it wound up looking like:

```python
  "\n          \n                    49 E. 95th St.             Chicago, IL 60628    \n          \n          "
```

Now we can normalize the data. We'd like to turn this into a simple two line address like:

```python
  "49 E. 95th St.\nChicago, IL 60628"
```

You can use whichever built-in string functions like `s.strip()`, `s.replace()`, and `s.split()` you see fit.

You may also use a regular expression here if you know how, but do not need to.

### Tokenization

For each record we collected, we will now split the collected text into tokens for search.

```json
{
  "name": "Example Park",
  "description": "This is a nice green park.",
  "history": "This park was founded in 1970."
  // other keys not shown
}
```

We'll take the "name", "history" and "description" text and convert them to a list of strings named "tokens".

The rules we will use for this are as follows:

1) Split the string on whitespace into individual "words."
For the purposes of this assignment, we will define a word to be a string that consists solely of letters, digits, and/or underscores (`_`) dashes (`-`) and parenthesis `()`.

2) Be sure to remove any of the following punctuation marks: `!.,'"?:` from the string.  (You may use a regular expression if you know how, but it is not necessary.)

  (You do not need to modify any other characters, you can leave unicode characters like "\u00a0" "\u2013", etc. as they are.)

3) Our search will be case-insensitive, so we will store all tokens in lower case.  Remember you can convert a string to lower case with `s.lower()`.

4) Certain words that are very common will be excluded, these words are provided for you as INDEX_IGNORE.  Any words in that list should be excluded from the tokens.

5) Duplicate tokens should be removed, you can use a `set()` to do this, but be sure to return a `list` from `tokenize`.

So for the above (truncated) example:

```json
{
  "name": "Example Park",
  "description": "This is a nice, GREEN park.",
  "history": "This park was founded in 1970.",
  "tokens": ["example", "park", "nice", "green", "founded", "1970"]
  // other keys not shown
}
```

(The word GREEN is capitalized in this example to show that capitalization is normalized in the tokens.)

### Putting it Together

Finally, you'll complete the `clean` method in this file.

This method should use the other methods you wrote (and any helper methods you deem necessary) to clean the park data.

It should load `parks.json`, clean the parks using the two methods and write the cleaned data out to `normalized_parks.json`.

## Testing

You can run the tests provided in `tests/test_crawler.py` and `test/test_cleanup.py` to verify the functionality of your code.

Automated tests are not provided for `clean()` and `crawl()`,
but you should test those yourself by running your program.

## Conclusion

At this point, you should be able to run your code (this will take a few minutes):

```
$ python -m parks
Fetching https://scrapple.fly.dev/parks
Fetching https://scrapple.fly.dev/parks/1
Fetching https://scrapple.fly.dev/parks/2
Fetching https://scrapple.fly.dev/parks/3
Fetching https://scrapple.fly.dev/parks/4
Fetching https://scrapple.fly.dev/parks/5
Fetching https://scrapple.fly.dev/parks/6
Fetching https://scrapple.fly.dev/parks/7
Fetching https://scrapple.fly.dev/parks/8
Fetching https://scrapple.fly.dev/parks/9
Fetching https://scrapple.fly.dev/parks/10
Fetching https://scrapple.fly.dev/parks
Fetching https://scrapple.fly.dev/parks?page=2
... (pages truncated)

Writing parks.json...
Cleaning parks.json and writing to normalized_parks.json
```

You should take a look at your output to make sure it matches what you expect.

**When you're ready to submit, add your `parks.json` and `normalized_parks.json` files to your repository and push them to GitHub.**

### Example Output

If you're wondering what your `normalized_parks.json` should look like, here is an example record:

```json
{
  "name": "Abbott (Robert) Park",
  "address": "49 E. 95th St. Chicago, IL 60628",
  "description": "\n        \n        Located in the Roseland Community Area, Abbott Park totals 25\u00a0acres and features a multi-purpose room and game room. Outside, the park offers four baseball diamonds, basketball, track and tennis courts, swimming pool, and two sprinklers. Many of these spaces are available for rental including our multi-purpose room and game room.Park-goers can participate in seasonal sports,\u00a0cheerleading,\u00a0aerobics, senior and teen clubs. On the cultural side, Abbott Park offers dance, music and movement. After school programs are offered throughout the school year, and during the summer youth can attend the Park District\u2019s popular six-week day camp.Specialty camps, including Sports Camp, are also offered in the summer.In addition to programs, Abbott Park hosts fun special events throughout the year for the entire family including\u00a0holiday events.\u00a0 \n        \n        ",
  "history": "\n        \n        The Chicago Park District acquired the site of Abbott Park as part of a ten-year plan to increase recreational opportunities in under-served neighborhoods in post-World War II Chicago. \u00a0In 1947, the Citizens Advisory Committee on Park Sites recommended the creation of a park to serve the rapidly growing African-American community near 95th Street and Michigan Avenue. \u00a0The Park District purchased the property southeast of that intersection in 1949 and 1950, and built a swimming pool and recreational facility the following year. \u00a0In 1956, the District sold a portion of the parkland to the Board of Education for use as Harlan High School.Robert Sengstacke Abbott (1868-1940), for whom the park is named, founded the influential Chicago Defender in 1905. \u00a0Born in Georgia, Abbott received his education in southern schools, and graduated from Chicago's Kent College of Law. \u00a0He was the only African-American in the class of 1899. \u00a0Abbott's lofty goal was to eliminate racial prejudice through his newspaper. \u00a0To promote racial equality, Abbott and his Chicago Defender newspaper urged southern blacks to migrate to Chicago and other northern cities for greater economic opportunity. \u00a0By 1918, the influential newspaper had a national circulation of 125,000, making it the largest-selling black newspaper in the country. \u00a0President of his Abbott Publishing Company, Abbott was also active in civic affairs.\u00a0 He served on Governor Frank O. Lowden's Race Relations Committee in 1919, on the Board of Commissioners of the Chicago World's Fair in 1934, and on the boards of the Art Institute, the Field Museum, and the Chicago Historical Society. \n        \n        ",
  "url": "https://scrapple.fly.dev/parks/1",
  "tokens": [
    "post-world", "during", "site", "1956", "aerobics", "president", "basketball", "district\u2019s", "michigan", "courts", "southeast", "race", "art", "under-served", "built", "multi-purpose", "economic", "camp", "campspecialty", "cultural", "1949", "increase", "growing", "pool", "roompark-goers", "seasonal", "holiday", "day", "board", "worlds", "after", "national", "frank", "boards", "facility", "available", "opportunity", "community", "urged", "civic", "influential", "chicago", "acquired", "area", "ten-year", "historical", "located", "defender", "neighborhoods", "class", "fair", "tennis", "citizens", "his", "portion", "high", "greater", "through", "schools", "northern", "harlan", "family", "summerin", "opportunities", "plan", "institute", "use", "diamonds", "(1868-1940)", "rapidly", "camps", "other", "georgia", "baseball", "committee", "game", "recreational", "founded", "sports", "near", "year", "prejudice", "participate", "received", "parkland", "senior", "black", "popular", "governor", "dance", "publishing", "roseland", "track", "special", "offered", "named", "served", "swimming", "125000", "sprinklers", "25", "1947", "clubs", "whom", "hosts", "field", "company", "part", "purchased", "property", "chicagos", "cities", "following", "circulation", "side", "museum", "features", "youth", "spaces", "education", "two", "affairs", "1919", "goal", "95th", "active", "serve", "had", "recommended", "teen", "movement", "creation", "war", "acres", "society", "including", "ii", "six-week", "1950", "outside", "totals", "avenue", "(robert)", "1918", "many", "rental", "district", "street", "racial", "kent", "making", "commissioners", "school", "1905", "lofty", "born", "relations", "room", "addition", "sold", "four", "blacks", "our", "entire", "o", "summer", "these", "country", "cheerleading", "throughout", "abbotts", "abbott", "1934", "events", "newspaper", "schoolrobert", "fun", "sites", "african-american", "eliminate", "southern", "lowdens", "promote", "advisory", "music", "can", "attend", "sengstacke", "college", "law", "programs", "equality", "1899", "graduated", "largest-selling", "migrate", "intersection", "offers", "only", "also" 
    ]
}
```