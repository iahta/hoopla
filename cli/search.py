import json

def searchTitle(arg):
    result = []
    with open("./data/movies.json") as f:
        movies_dict = json.load(f)
    
    search_arg = arg.lower()
    for movie in movies_dict.get("movies", []):
        title = movie.get("title", "")
        title_search = title.lower()
        if search_arg in title_search:
            result.append(title)
    result = result[:5]
    for i in range(len(result)):
        print(f"{i + 1}. {result[i]}")
