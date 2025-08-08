# The AI Dungeon Master's Command Center

The AI Dungeon Master's Command Center is a comprehensive, all-in-one desktop application designed to augment the tabletop role-playing game (TTRPG) experience.

## Quick Start Guide (Running from Source)

This guide explains how to run the application directly from its source code. This is the recommended method for running the app.

### 1. Prerequisites
* You must have Python 3.9 or newer installed.
* You need Git to clone the repository.

### 2. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/your-username/DM-Command-Center-Desktop.git
cd DM-Command-Center-Desktop
```
### 3. Create a Virtual Environment
Create a dedicated virtual environment for the project. This keeps its dependencies separate from your other Python projects.

```bash
# On Windows
python -m venv .venv

# On macOS/Linux
python3 -m venv .venv
```
### 4. Activate the Environment
You must activate the environment before installing packages.

```bash
# On Windows
.\.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```
### 5. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```
### 6. Set Up Your API Key
The application requires an OpenAI API key.

Create a file in the project's root directory named `.env`.

Add your API key to this file in the following format:
```
OPENAI_API_KEY="sk-YourSecretKeyGoesHere"
```
### 7. Run the Application
You're all set! Launch the application by running the `app.py` script.

```bash
python app.py
```
The application window should now appear. The first time you run it, it may take a minute to build the local rules database.
