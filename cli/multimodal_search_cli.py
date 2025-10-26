import argparse


from lib.multimodal_search import (
    verify_image_embedding,
    image_search_command,
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Serach CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="verify image embedding works")
    verify_parser.add_argument("path", type=str, help="Path of image to verify")

    image_search_parser = subparsers.add_parser("image_search", help="Search movies using image")
    image_search_parser.add_argument("path", type=str, help="Path of image to search")
    

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            results = image_search_command(args.path)
            for i, res in enumerate(results):
                print(f"{i+1}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"       {res['description'][:100]}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()