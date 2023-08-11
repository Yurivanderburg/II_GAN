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
        #start_date = "1995-06-16"
        start_date = "2023-03-23"
        url = f"https://api.nasa.gov/planetary/apod?api_key=DB9DfUXnW0fZN4knCcghKyZ3f1GeKqTXkeymAhIr&start_date={start_date}&end_date={today}"

        all_results = requests.get(url).json()

        # Check whether the words "galaxy" and "spiral" are in the description
        for result in all_results:
            date = result['date']
            if ("spiral" and "galaxy") in result['explanation']:
                print(f"APOD Image of {date} contains a spiral galaxy!")

                # Download images
                image = requests.get(result['url']) # Change to hdurl if you want a high-resolution picture ;)
                with open(f"/home/yuri/Documents/Uni/Master/Semester3/Internship/Data/SpiralGalaxy/apod_{date}.jpg", "wb") as f:
                    f.write(image.content)


    except:
        return "URL Error"  # Return reload icon


if __name__ == "__main__":
    main()
