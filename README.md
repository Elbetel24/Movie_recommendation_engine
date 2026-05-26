# Movie Recommendation Engine

A simple movie recommendation tool built with Python and cosine similarity.

## Features

- Reads movie data from `movie_dataset.csv`
- Combines movie features like `keywords`, `cast`, `genres`, and `director`
- Builds a content-based recommendation model using `CountVectorizer`
- Computes movie similarity using cosine similarity
- Accepts user input for movies they have watched
- Recommends similar movies based on the combined watched movie list
- Excludes watched titles from recommendations

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

## Setup

1. Create a Python virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/Scripts/activate   # Windows
   source venv/bin/activate      # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. Make sure `movie_dataset.csv` is in the same folder as `movie_recommender.py`.

## Usage

Run the recommender from the project directory:

```bash
python movie_recommender.py
```

When prompted, enter one or more movie titles you have watched, separated by commas.

Example:

```text
Enter movies you've watched (comma-separated): Avatar, Titanic
```

The script will then print a ranked list of similar movies.

## Notes

- Movie titles must match the dataset entries exactly.
- If a title is not found, a warning is shown and recommendations are generated from the remaining valid movies.
- Future improvements can include fuzzy title matching and a better user interface.
