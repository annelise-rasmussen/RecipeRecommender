from flask import Flask, render_template, request
from markupsafe import Markup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import openai
import configparser
import mysql.connector


# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the OpenAI API key from the config file
openai.api_key = config.get('openai', 'api_key')

# Load the recipes dataset
recipes_df = pd.read_csv('cleaned_recipes1.csv')

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Create the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(recipes_df['recipe_text'])

db_config = {
    'host': config['mysql']['host'],
    'user': config['mysql']['user'],
    'password': config['mysql']['password'],
    'database': config['mysql']['database']
}

# Database connection function
def get_db_connection():
    return mysql.connector.connect(**db_config)

app = Flask(__name__)

@app.route('/')
def home():
    titleText = "Gluten Free Recipe Favs!"
    bodyText = Markup("""
        <br>
        <h1>We Love All Things Gluten Free</h1>
                      <br>
        <h3> Don't give up on having tasty gluten free food. These recipes are AMAZING and SIMPLE.
                      Visit my favorite websites and recipes. <br>Do you need help coming up with meal ideas? 
                      Try the recommender on the recipe search tab. <br> Do you need advice on substitutions for different allergies? Check out the ingredient substitutions tab. 
                      <br>Happy cooking and baking :)
        
       
    """)
    return render_template('template.html', titleText=titleText, bodyText=bodyText)

@app.route('/fav_websites')
def fav_websites():
    titleText = "Favorite Gluten Free Websites"
    bodyText = Markup("""
        <br>
        <h2>Here are a few websites that I use the most often:</h2>
        <br>
	<br>
	<ul> <h4>
            <li><a href="https://theloopywhisk.com/" target="_blank">The Loopy Whisk</a></li>
            <li><a href="https://www.mamaknowsglutenfree.com/" target="_blank">Mama Knows Gluten Free</a></li>
            <li><a href="https://meaningfuleats.com/" target="_blank">Meaningful Eats</a></li>
                      </h4>
        </ul>
    """)
    return render_template('template.html', titleText=titleText, bodyText=bodyText)

@app.route('/fav_recipes')
def fav_recipes():
    titleText = "Favorite Recipes"

    # Fetch recipe titles from the database
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, title FROM recipes;")
    recipes = cursor.fetchall()
    connection.close()

    # Dynamically generate links for recipes
    recipe_links = "".join(
        f'<li><a href="/recipe/{recipe[0]}">{recipe[1]}</a></li>' for recipe in recipes
    )

    bodyText = Markup(f"""
    <div class="recipes-section">
        <h1>Favorite Recipes:</h1>
        <br>
        <div class="list-container">
            <ul> <h4>
                {recipe_links}
            </h4>
            </ul>
        </div>
    </div>
    </body>
    </html>
    """)

    return render_template('template.html', titleText=titleText, bodyText=bodyText)

@app.route('/recipe/<int:recipe_id>')
def show_recipe(recipe_id):
    # Fetch recipe details from the database
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Fetch recipe title
    cursor.execute("SELECT title FROM recipes WHERE id = %s;", (recipe_id,))
    recipe = cursor.fetchone()
    
    if not recipe:
        connection.close()
        return "Recipe not found", 404

    # Fetch ingredients
    cursor.execute("SELECT ingredient FROM ingredients WHERE recipe_id = %s;", (recipe_id,))
    ingredients = [row["ingredient"] for row in cursor.fetchall()]

    # Fetch instructions
    cursor.execute("SELECT step_number, instruction FROM instructions WHERE recipe_id = %s ORDER BY step_number;", (recipe_id,))
    instructions = [row["instruction"] for row in cursor.fetchall()]
    connection.close()

    return render_template(
        'recipe.html',
        titleText=recipe["title"],
        ingredients=ingredients,
        instructions=instructions
    )



@app.route('/Recipe_Recommender', methods=['GET', 'POST'])
def recipe_search():
    query = request.args.get('query', '').strip()
    selected_cuisine = request.args.get('cuisine', '')
    selected_complexity = request.args.get('complexity', '')
    results = []
    recommendations = []
    show_no_results = False

    # Map complexity categories
    complexity_categories = {"Low": (1, 3), "Medium": (4, 6), "High": (7, 10)}

    # Get unique cuisines for dropdown options
    cuisines = sorted(recipes_df['cuisine'].dropna().unique())

    if query or selected_cuisine or selected_complexity:
        # Apply query filter
        mask = recipes_df['recipe_text'].str.contains(query, case=False, na=False) if query else pd.Series(True, index=recipes_df.index)

        # Apply cuisine filter
        if selected_cuisine:
            mask &= recipes_df['cuisine'] == selected_cuisine

        # Apply complexity filter based on selected category
        if selected_complexity and selected_complexity in complexity_categories:
            low, high = complexity_categories[selected_complexity]
            mask &= recipes_df['complexity_normalized'].between(low, high)

        # Filter recipes based on the combined mask
        filtered_recipes = recipes_df[mask]

        # Convert search results to dictionary
        results = filtered_recipes[['Title', 'Source_URL']].to_dict(orient='records')

        # Compute recommendations based on similarity
        if query:
            input_vector = tfidf.transform([query])
            scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
            recipes_with_scores = recipes_df.copy()
            recipes_with_scores['similarity'] = scores
            recommendations = recipes_with_scores.sort_values(by='similarity', ascending=False).head(5).to_dict(orient='records')

        # Show "No results found" message if no results
        show_no_results = not results

    return render_template(
        'recipe_recommender.html',
        titleText="Recipe Search Results",
        results=results,
        recommendations=recommendations,
        query=query,
        cuisines=cuisines,
        complexity_scores=["Low", "Medium", "High"],  # Pass categories instead of numbers
        selected_cuisine=selected_cuisine,
        selected_complexity=selected_complexity,
        show_no_results=show_no_results,
    )

@app.route('/Ingredient_Substitutions', methods=['GET', 'POST'])
def ingredient_substitutions():
    query = request.form.get('query', '').strip()
    openai_response = None

    if query:
    
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are an expert in cooking and nutrition."},
            {"role": "user", "content": f"Provide advice on ingredient substitutions: {query}"}
                ],
                max_tokens=150,
                temperature=0.7,
            )
        openai_response = response['choices'][0]['message']['content'].strip()

    return render_template(
        'ingredient_substitutions.html',
        titleText="Ingredient Substitutions for Dairy Products",
        openai_response=openai_response,
        query=query
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
