import argparse


from lib.multimodal_search import (
    verify_image_embedding,
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Serach CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Search movies using BM25")
    verify_parser.add_argument("path", type=str, help="Path of image to verify")
    

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case _:
            parser.print_help()
        




if __name__ == "__main__":
    main()