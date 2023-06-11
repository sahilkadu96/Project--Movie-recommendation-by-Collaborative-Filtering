from pydantic import BaseModel, Field

class MovieRecommend():
    movie_name: str
    num_of_movies_to_be_recommended: int
    def __init__(self, movie_name, num_of_movies_to_be_recommended):
        self.movie_name = movie_name
        self.num_of_movies_to_be_recommended = num_of_movies_to_be_recommended


class MovieRecommendRequest(BaseModel):
    movie_name: str = Field()
    num_of_movies_to_be_recommended: int = Field()

