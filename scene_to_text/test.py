from GPT4o_API_vision import Images, Video

# IMPORTANT: for local images and videos, write the entire local path of the file(s)
# test
# urls
urls = [
  "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
  "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
]
# local image files
image_paths = ["ENTER PATH/dj_cooky.jpeg"]
YOUR_API_KEY = "ENTER API KEY"
imageObj = Images(urls, "Compare and contrast these two pictures if you see two images. If you see only one image, describe that image.", image_paths, YOUR_API_KEY)
print("\nImages test. Urls.")
print(imageObj.url_images_prompt())
print("\nImages test. Local files.")
print(imageObj.local_images_prompt())

# video file, local
f = "ENTER PATH/bison.mp4"
videoObj = Video(f, "Describe what's happening in this short video.", YOUR_API_KEY)
print("\nVideo test.")
print(videoObj.video_prompt())