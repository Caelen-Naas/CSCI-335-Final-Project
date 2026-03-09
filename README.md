Initial commit outta the way!
Added readme, git ignore, requirements.txt and .env
Please update the .gitignore by uncommenting .env when you clone/pull

.env and data\ folder aren't relevant if we don't use IMDB dataset/API
Also recomment creating a virtual environment to work in, I'm using python 3.12

#Create venv with py 3.12
py -V:3.12 -m venv .venv 

#get requirements
pip install -r requirements.txt

#Update requirements when adding libraries
pip freeze > requirements.txt