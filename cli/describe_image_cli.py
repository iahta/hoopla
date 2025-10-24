import argparse
import mimetypes

from lib.search_utils import (
    load_image,
)

from gemini import multimodal_results

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--image", type=str, help="Image you want desribed")
    parser.add_argument("--query", type=str, help="What question you want answered")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    if mime == None:
        print("Missing or Incorrect Image file type")
        return
    query_image = load_image(args.image)
    multimodal_results(args.query, query_image, mime)




if __name__ == "__main__":
    main()