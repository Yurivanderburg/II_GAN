import requests
import json
import datetime
import numpy as np



def main():
    try:
        # Get data from apod-api
        # Only today:
        today = datetime.date.today()
        url_today = 'https://api.nasa.gov/planetary/apod?api_key=DB9DfUXnW0fZN4knCcghKyZ3f1GeKqTXkeymAhIr'

        # For a timeframe

        years = np.arange(2012, 2021, 1)
        #start_date = "1995-06-16"

        # Loop over the years, so we only have 365 entries in each interation
        for year in years:
            image_counter = 0
            start_date = f"{year}-03-11"
            end_date = f"{year}-12-31"

            url = f"https://api.nasa.gov/planetary/apod?api_key=DB9DfUXnW0fZN4knCcghKyZ3f1GeKqTXkeymAhIr&start_date=" \
                  f"{start_date}&end_date={end_date}"

            all_results = requests.get(url).json()

            # Check whether the words "galaxy" and "spiral" are in the description
            for result in all_results:
                date = result['date']
                if ("spiral" and "galaxy") in result['explanation']:
                    image_counter += 1

                    # Download images
                    image = requests.get(result['url'])  # Change to hdurl if you want a high-resolution picture ;)
                    with open(f"/home/yuri/Documents/Uni/Master/Semester3/Internship/Data/SpiralGalaxy/apod_{date}.jpg", "wb") as f:
                        f.write(image.content)
            print(f"Downloaded {image_counter} images from APOD in {year}")

    except:
        return "URL Error"


if __name__ == "__main__":
    main()
