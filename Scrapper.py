#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class WebDataScrapper:
    def __init__(self, years, url):
        self.years = years
        self.url = url
        self.session = requests.Session()
        self.all_match_data = []
        
    def get_soup(self, url):
        """
        Request a URL and create a BeautifulSoup object from the response text.

        Args:
            url (str): URL of the webpage.

        Returns:
            BeautifulSoup: BeautifulSoup object of the webpage.
        """
        page_data = self.session.get(url)
        return BeautifulSoup(page_data.text, 'html.parser')


    def get_team_links(self, league_table):
        """
        Extract team links from the league table.

        Args:
            league_table (Tag): BeautifulSoup Tag object containing the league table.

        Returns:
            list: List of team links in the league table.
        """
        team_links = []
        for link in league_table.find_all('a'):
            href = link.get("href")
            if '/squads/' in href:
                team_links.append(href)
        return team_links


    def get_shooting_links(self, soup):
        """
        Extract shooting links from a BeautifulSoup object.

        Args:
            soup (BeautifulSoup): BeautifulSoup object of the webpage.

        Returns:
            list: List of shooting links in the BeautifulSoup object.
        """
        shooting_links = []
        for link in soup.find_all('a'):
            href = link.get("href")
            if href and 'all_comps/shooting/' in href:
                shooting_links.append(href)
        return shooting_links


    def merge_data(self, matches, shooting):
        """
        Merge matches and shooting data.

        Args:
            matches (DataFrame): Pandas DataFrame containing match data.
            shooting (DataFrame): Pandas DataFrame containing shooting data.

        Returns:
            DataFrame: Merged DataFrame with relevant columns.
        """

        try:
            merged_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            return None
        return merged_data[merged_data["Comp"] == "Premier League"]
    
    def extract_data(self):
        # Iterate through the years
        for year in self.years:
            time.sleep(5)
            soup = self.get_soup(self.url)
            league_table = soup.select('table.stats_table')[0]

            # Get the team links from the league table
            team_links = self.get_team_links(league_table)
            full_team_urls = [f"https://fbref.com{link}" for link in team_links]

            # Update the URL for the previous season
            previous_season_link = soup.select("a.prev")[0].get("href")
            self.url = f"https://fbref.com{previous_season_link}"

            # Iterate through the team URLs
            for team_url in full_team_urls:
                time.sleep(5)
                team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
                data = self.session.get(team_url)
                matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
                soup = self.get_soup(team_url)

                # Get the shooting links from the team page
                shooting_links = self.get_shooting_links(soup)
                data = self.session.get(f"https://fbref.com{shooting_links[0]}")
                shooting = pd.read_html(data.text, match="Shooting")[0]
                shooting.columns = shooting.columns.droplevel()

                # Merge match and shooting data
                team_data = self.merge_data(matches, shooting)

            if team_data is not None:
                team_data["Season"] = year
                team_data["Team"] = team_name
                self.all_match_data.append(team_data)
            time.sleep(5)

        # Combine all the match data into a single DataFrame
        match_df = pd.concat(self.all_match_data)
        match_df.to_csv("matches.csv")
        return match_df


# In[5]:


if __name__ == "__main__":
    years = [2022, 2021, 2020]
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    extractor = WebDataScrapper(years, url)
    data = extractor.extract_data()
    print(data.head())  


# In[ ]:




