import requests
import json
import datetime

# Params
start_date = "1995-06-26"
end_date = "1995-12-31"
#end_date = datetime.date.today()

# Edit this to search for different images
required_term1 = "spiral"
required_term2 = "galaxy"

do_download = True

def main():
    """
    Python script that grabs several "Astronomy Pictures of the Day" (APOD), using APIs.
    Can specify keywords which must be contained in the description of the image
    Note: If the timescale is large (> 2 years), it would be wise to split the url, otherwise it might break.
    """
    try:
        #
        url = f"https://api.nasa.gov/planetary/apod?api_key=DB9DfUXnW0fZN4knCcghKyZ3f1GeKqTXkeymAhIr&start_date=" \
              f"{start_date}&end_date={end_date}"

        all_results = requests.get(url).json()
        image_counter = 0 # Counts the images

        # Check whether the words are in the description
        for result in all_results:
            date = result['date']

            if (required_term1 and required_term2) in result['explanation']:
                if result['media_type'] == "image":

                    image_counter += 1

                    # Download images
                    if do_download:
                        image = requests.get(result['url'])  # Change to hdurl if you want a high-resolution picture ;)
                        with open(f"Data/SpiralGalaxy/apod_{date}.jpg", "wb") as f:
                            f.write(image.content)

            if do_download:
                print(f"Downloaded {image_counter} images between {start_date} and {end_date} matching the description.")
            else:
                print(f"Found {image_counter} images between {start_date} and {end_date} matching the description. "
                      f"They were not downloaded.")

    except:
        return "URL Error"


if __name__ == "__main__":
    main()
