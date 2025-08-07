import os
import csv
from customtkinter.windows.widgets.ctk_input_dialog import CTkInputDialog
import customtkinter as ctk
import threading
import queue
import speech_recognition as sr
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from tkinter import filedialog
from dotenv import load_dotenv, get_key, set_key
from PIL import Image
import requests
import io
from openai import OpenAI
import database

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# --- ENCOUNTER CALCULATION DATA ---

# XP Thresholds by Character Level
XP_THRESHOLDS = {
    1: {"easy": 25, "medium": 50, "hard": 75, "deadly": 100},
    2: {"easy": 50, "medium": 100, "hard": 150, "deadly": 200},
    3: {"easy": 75, "medium": 150, "hard": 225, "deadly": 400},
    4: {"easy": 125, "medium": 250, "hard": 375, "deadly": 500},
    5: {"easy": 250, "medium": 500, "hard": 750, "deadly": 1100},
    # You can expand this list to include all 20 levels
}

# Encounter XP Multipliers by number of monsters
ENCOUNTER_MULTIPLIERS = {
    1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0,
    7: 2.5, 8: 2.5, 9: 2.5, 10: 2.5,
    11: 3.0, 12: 3.0, 13: 3.0, 14: 3.0,
    15: 4.0
    # A multiplier of x4 is used for 15 or more monsters
}

# --- AUDIO KEYWORD MAPPING ---
AUDIO_TRIGGERS = {
    "music_combat": ["fight", "combat", "attack", "damage", "hit", "sword", "arrow", "axe", "initiative", "roll for damage"],
    "music_tavern": ["tavern", "inn", "bar", "ale", "beer", "quest", "barkeep", "gossip"],
    "music_tense": ["sneak", "stealth", "trap", "danger", "creeping", "shadows", "darkness", "eerie"],
    "sfx_sword": ["sword hits", "clash of steel", "parry"],
}

AUDIO_FILES = {
    "music_combat": "audio/music/454_Broken_Pantheon.mp3",
    "music_tavern": "audio/ambiance/446_Between_Adventures.mp3",
    "music_tense": "audio/music/212_Witch_Mountain.mp3",
    "sfx_sword": "audio/sfx/sword-hit-7160.mp3",
}

# --- AI INITIALIZATION ---
def initialize_ai_rules_lawyer():
    """
    Initializes the AI Rules Lawyer chatbot. It checks if a local vector store exists.
    If not, it creates one from the D&D SRD PDF and saves it for future use.
    """
    print("Initializing AI Rules Lawyer...")
    
    persist_directory = "db_chroma"
    embeddings = OpenAIEmbeddings(chunk_size=200) # Batch requests to avoid API errors

    if os.path.exists(persist_directory):
        print("Loading existing database...")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Creating new database... (This is a one-time process)")
        data_path = "data/dnd_srd.pdf"
        loader = PyPDFLoader(data_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            docs, 
            embeddings, 
            persist_directory=persist_directory
        )
        print("Database created and saved.")

    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    print("AI Rules Lawyer Initialized.")
    return chain

def initialize_world_forge():
    print("Initializing World Forge...")
    llm = ChatOpenAI(model_name="gpt-4o")
    print("World Forge Initialized.")
    return llm

# --- MAIN DESKTOP APPLICATION CLASS ---
class DMCommandCenterApp(ctk.CTk):
    def __init__(self, rules_lawyer_chain, world_forge_llm):
        super().__init__()

        # --- Campaign Management Bar ---
        campaign_bar_frame = ctk.CTkFrame(self, height=50)
        campaign_bar_frame.pack(side="top", fill="x", padx=10, pady=(10, 0))

        new_campaign_button = ctk.CTkButton(campaign_bar_frame, text="New Campaign", command=self.new_campaign)
        new_campaign_button.pack(side="left", padx=10, pady=10)

        open_campaign_button = ctk.CTkButton(campaign_bar_frame, text="Open Campaign", command=self.open_campaign)
        open_campaign_button.pack(side="left", padx=10, pady=10)

        self.campaign_name_label = ctk.CTkLabel(campaign_bar_frame, text="Campaign: None", font=ctk.CTkFont(weight="bold"))
        self.campaign_name_label.pack(side="left", padx=10, pady=10)


        # --- AI Model & Data Storage ---
        self.rules_lawyer_chain = rules_lawyer_chain
        self.world_forge_llm = world_forge_llm
        self.rules_chat_history = []
        self.monsters = self.load_monsters()

        # --- Ambiance Engine State ---
        self.is_listening = False
        self.listener_thread = None
        self.ambiance_queue = queue.Queue()
        self.selected_mic_index = 0
        self.audio_thread = None
        self.current_music_mood = None
        self.stop_audio_flag = threading.Event()

        # --- AI Clients ---
        self.openai_client = OpenAI()

        # --- Map Forge State ---
        self.map_image_data = None

        # --- World Forge State ---
        self.portrait_image_data = None

        # --- Campaign State ---
        self.current_campaign_path = None
        self.campaign_name_label = None

        # --- Window Configuration ---
        self.title("AI Dungeon Master's Command Center")
        self.geometry("800x650")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # --- Create Tabs ---
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(padx=20, pady=20, fill="both", expand=True)
        self.tab_view.add("World Forge")
        self.tab_view.add("AI Rules Lawyer")
        self.tab_view.add("Encounter Architect")
        self.tab_view.add("Ambiance Engine")
        self.tab_view.add("Map Forge")
        self.tab_view.add("Campaign Explorer")
        self.tab_view.add("Settings")
        
        # --- Configure Tabs ---
        self.setup_world_forge_tab()
        self.setup_rules_lawyer_tab()
        self.setup_encounter_architect_tab()
        self.setup_ambiance_tab()
        self.setup_map_forge_tab()
        self.setup_campaign_explorer_tab()
        self.setup_settings_tab()

        # --- Start Queue Processor ---
        self.process_ambiance_queue()

    def process_ambiance_queue(self):
        """Processes messages from the ambiance queue to update the UI safely."""
        try:
            msg_type, data = self.ambiance_queue.get_nowait()
            if msg_type == 'log':
                self.append_to_textbox(self.transcription_log, data + "\n")
            elif msg_type == 'portrait':
                self.portrait_image_data = data # Save for database
                image = Image.open(io.BytesIO(data))
                image.thumbnail((400, 400))
                ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
                self.portrait_label.configure(image=ctk_image, text="")
            elif msg_type == 'status':
                self.ambiance_status_label.configure(text=f"Status: {data}")
            elif msg_type == 'mood':
                self.ambiance_mood_label.configure(text=f"Mood: {data}")
            elif msg_type == 'button_state':
                self.ambiance_button.configure(state=data)
            elif msg_type == 'button_text':
                self.ambiance_button.configure(text=data)
            elif msg_type == 'map':
                image = Image.open(io.BytesIO(data))
                self.map_image_data = data # Save raw bytes for saving
                image.thumbnail((800, 800)) # Larger thumbnail for map
                ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
                self.map_display_label.configure(image=ctk_image, text="")
                self.save_map_button.configure(state="normal")
                self.generate_map_button.configure(state="normal", text="Generate Map")
            elif msg_type == 'suggestion':
                self.update_textbox(self.suggestion_output, data)
                self.suggestion_button.configure(state="normal", text="Get AI Suggestions")
        except queue.Empty:
            pass
        finally:
            # Check again after 100ms
            self.after(100, self.process_ambiance_queue)

    def toggle_listening(self):
        """Starts or stops the ambiance engine listening thread."""
        if self.is_listening:
            self.is_listening = False
            self.ambiance_queue.put(('status', "Stopping..."))
            self.ambiance_queue.put(('button_state', "disabled"))
        else:
            self.is_listening = True
            self.listener_thread = threading.Thread(target=self.run_ambiance_engine)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            self.ambiance_queue.put(('status', "Listening..."))
            self.ambiance_queue.put(('button_text', "Stop Listening"))

    def run_ambiance_engine(self):
        """
        The core function for the ambiance engine.
        Runs in a background thread, listens for audio, and transcribes it.
        """
        # Log the list of microphones found
        try:
            mic_list = sr.Microphone.list_microphone_names()
            self.ambiance_queue.put(('log', f"Microphones found: {mic_list}"))
        except Exception as e:
            self.ambiance_queue.put(('log', f"Error getting microphone list: {e}"))

        # Check for microphone availability first
        try:
            if not sr.Microphone.list_microphone_names():
                self.ambiance_queue.put(('log', "ERROR: No microphone found on this system."))
                self.is_listening = False
        except Exception as e:
            self.ambiance_queue.put(('log', f"ERROR: Could not check for microphones. {e}"))
            self.is_listening = False

        if not self.is_listening:
            self.ambiance_queue.put(('status', "Error"))
            self.ambiance_queue.put(('button_text', "Start Listening"))
            self.ambiance_queue.put(('button_state', "normal"))
            return

        r = sr.Recognizer()
        mic = sr.Microphone(device_index=self.selected_mic_index)

        try:
            with mic as source:
                self.ambiance_queue.put(('log', "Calibrating for ambient noise..."))
                r.adjust_for_ambient_noise(source)
                self.ambiance_queue.put(('log', "Calibration complete. Listening..."))

            while self.is_listening:
                try:
                    with mic as source:
                        audio = r.listen(source)

                    text = r.recognize_google(audio)
                    self.ambiance_queue.put(('log', f"Heard: {text}"))

                    # Analyze text and trigger audio
                    trigger = self.analyze_text_for_audio(text)
                    if trigger:
                        self.trigger_audio(trigger)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    self.ambiance_queue.put(('log', f"API error: {e}"))
                    self.is_listening = False
                except Exception as e:
                    self.ambiance_queue.put(('log', f"An audio error occurred: {e}"))
                    self.is_listening = False
        except Exception as e:
            self.ambiance_queue.put(('log', f"ERROR: Failed to open microphone. Is it in use? Details: {e}"))
            self.is_listening = False

        # Thread finished, update UI state via the queue
        self.ambiance_queue.put(('status', "Idle"))
        self.ambiance_queue.put(('button_text', "Start Listening"))
        self.ambiance_queue.put(('button_state', "normal"))
        self.ambiance_queue.put(('log', "Listener stopped."))

    def setup_map_forge_tab(self):
        """Creates the widgets for the Map Forge tab."""
        tab = self.tab_view.tab("Map Forge")

        # Configure a 2-column grid
        tab.grid_columnconfigure(0, weight=1) # Controls
        tab.grid_columnconfigure(1, weight=2) # Image display
        tab.grid_rowconfigure(0, weight=1)

        # --- Left Column: Controls ---
        left_frame = ctk.CTkFrame(tab, fg_color="transparent")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1) # Allow textbox to expand

        prompt_label = ctk.CTkLabel(left_frame, text="Describe the map you want to create:", font=ctk.CTkFont(weight="bold"))
        prompt_label.grid(row=0, column=0, padx=10, pady=(0,5), sticky="w")

        self.map_input = ctk.CTkTextbox(left_frame, height=150)
        self.map_input.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.map_input.insert("0.0", "A small, ruined keep on a cliff overlooking the sea.")

        button_frame = ctk.CTkFrame(left_frame)
        button_frame.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        self.generate_map_button = ctk.CTkButton(button_frame, text="Generate Map", command=self.start_map_generation_thread)
        self.generate_map_button.pack(side="right", padx=(5, 0))

        self.save_map_button = ctk.CTkButton(button_frame, text="Save Map", command=self.save_map, state="disabled")
        self.save_map_button.pack(side="right", padx=(0, 5))

        # --- Right Column: Image Display ---
        right_frame = ctk.CTkFrame(tab)
        right_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        self.map_display_label = ctk.CTkLabel(right_frame, text="Generated map will appear here.", text_color="gray")
        self.map_display_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def start_map_generation_thread(self):
        """Starts the map generation in a new thread."""
        self.generate_map_button.configure(state="disabled", text="Generating...")
        self.save_map_button.configure(state="disabled") # Disable save button during generation
        thread = threading.Thread(target=self.generate_map)
        thread.daemon = True
        thread.start()

    def generate_map(self):
        """Constructs a prompt and calls the DALL-E API to generate a map."""
        try:
            user_description = self.map_input.get("0.0", "end-1c")
            if not user_description.strip():
                print("Map description is empty.")
                self.generate_map_button.configure(state="normal", text="Generate Map")
                return

            prompt = (
                f"A top-down, 2D, black and white battle map for a tabletop role-playing game like Dungeons and Dragons. "
                f"The style should be clean and clear line art, suitable for printing. "
                f"The map should depict: {user_description}. "
                "Do not include a grid, text, icons, or any other distracting elements."
            )

            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard"
            )
            image_url = response.data[0].url

            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_data = image_response.content
            self.ambiance_queue.put(('map', image_data))

        except Exception as e:
            print(f"Error generating map: {e}")
            self.ambiance_queue.put(('log', f"ERROR: Could not generate map. {e}"))
            self.generate_map_button.configure(state="normal", text="Generate Map")

    def save_map(self):
        """Saves the currently displayed map image to a file."""
        if self.map_image_data is None:
            print("No map image data to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
            title="Save Map Image"
        )

        if filepath:
            try:
                with open(filepath, "wb") as f: # Open in write-bytes mode
                    f.write(self.map_image_data)
                print(f"Map saved to {filepath}")
            except Exception as e:
                print(f"Error saving map: {e}")

    def new_campaign(self):
        """Creates a new campaign database file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("Campaign Database", "*.db"), ("All Files", "*.*")],
            title="Create New Campaign"
        )
        if filepath:
            database.init_db(filepath)
            self.current_campaign_path = filepath
            campaign_name = os.path.basename(filepath)
            self.campaign_name_label.configure(text=f"Campaign: {campaign_name}")
            # Here you would also enable/disable other UI elements

    def open_campaign(self):
        """Opens an existing campaign database file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Campaign Database", "*.db"), ("All Files", "*.*")],
            title="Open Campaign"
        )
        if filepath:
            self.current_campaign_path = filepath
            campaign_name = os.path.basename(filepath)
            self.campaign_name_label.configure(text=f"Campaign: {campaign_name}")
            self.populate_campaign_explorer()

    def new_campaign(self):
        """Creates a new campaign database file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("Campaign Database", "*.db"), ("All Files", "*.*")],
            title="Create New Campaign"
        )
        if filepath:
            database.init_db(filepath)
            self.current_campaign_path = filepath
            campaign_name = os.path.basename(filepath)
            self.campaign_name_label.configure(text=f"Campaign: {campaign_name}")
            self.populate_campaign_explorer()

    def setup_campaign_explorer_tab(self):
        """Creates the widgets for the Campaign Explorer tab."""
        tab = self.tab_view.tab("Campaign Explorer")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.campaign_explorer_frame = ctk.CTkScrollableFrame(tab, label_text="Saved NPCs")
        self.campaign_explorer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def populate_campaign_explorer(self):
        """Clears and repopulates the campaign explorer with items from the DB."""
        # Clear existing widgets
        for widget in self.campaign_explorer_frame.winfo_children():
            widget.destroy()

        # Populate with NPCs
        npcs = database.get_all_npcs(self.current_campaign_path)
        for npc_id, npc_name in npcs:
            npc_button = ctk.CTkButton(
                self.campaign_explorer_frame,
                text=npc_name,
                command=lambda npc_id=npc_id: self.load_npc(npc_id)
            )
            npc_button.pack(fill="x", padx=5, pady=2)

    def load_npc(self, npc_id):
        """Loads a specific NPC's data from the database into the World Forge."""
        try:
            conn = sqlite3.connect(self.current_campaign_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, description, portrait FROM npcs WHERE id = ?", (npc_id,))
            npc = cursor.fetchone()
            if npc:
                name, description, portrait_data = npc

                # Update World Forge UI
                self.update_textbox(self.forge_output, description)

                # Update portrait
                if portrait_data:
                    self.portrait_image_data = portrait_data
                    image = Image.open(io.BytesIO(portrait_data))
                    image.thumbnail((400, 400))
                    ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
                    self.portrait_label.configure(image=ctk_image, text="")
                else:
                    self.portrait_label.configure(image=None, text="No portrait available.")

                # Switch to the World Forge tab to show the loaded content
                self.tab_view.set("World Forge")

        except sqlite3.Error as e:
            print(f"Database error loading NPC: {e}")
        finally:
            if conn:
                conn.close()

    def setup_ambiance_tab(self):
        """Creates the widgets for the Dynamic Ambiance Engine tab."""
        tab = self.tab_view.tab("Ambiance Engine")

        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Control Frame
        control_frame = ctk.CTkFrame(tab)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.ambiance_button = ctk.CTkButton(control_frame, text="Start Listening", command=self.toggle_listening)
        self.ambiance_button.pack(side="left", padx=10, pady=10)

        self.ambiance_status_label = ctk.CTkLabel(control_frame, text="Status: Idle")
        self.ambiance_status_label.pack(side="left", padx=10, pady=10)

        self.ambiance_mood_label = ctk.CTkLabel(control_frame, text="Mood: -")
        self.ambiance_mood_label.pack(side="left", padx=10, pady=10)

        # Log Frame
        log_frame = ctk.CTkFrame(tab)
        log_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        log_label = ctk.CTkLabel(log_frame, text="Transcription Log", font=ctk.CTkFont(weight="bold"))
        log_label.pack(padx=10, pady=(10,0))

        self.transcription_log = ctk.CTkTextbox(log_frame, state="disabled")
        self.transcription_log.pack(padx=10, pady=10, fill="both", expand=True)
        
    def load_monsters(self):
        """Loads monster data from the CSV file."""
        monsters = {}
        try:
            with open("data/monsters.csv", "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    monsters[row["Name"]] = {"CR": row["CR"], "XP": int(row["XP"])}
            print(f"Loaded {len(monsters)} monsters.")
        except FileNotFoundError:
            print("Error: monsters.csv not found!")
        return monsters

    def setup_world_forge_tab(self):
        """Creates the widgets for the World Forge tab."""
        tab = self.tab_view.tab("World Forge")
        
        # Configure a 2-column grid
        tab.grid_columnconfigure(0, weight=1) # Text content
        tab.grid_columnconfigure(1, weight=1) # Image content
        tab.grid_rowconfigure(0, weight=1)

        # --- Left Column: Text Generation ---
        left_frame = ctk.CTkFrame(tab, fg_color="transparent")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(2, weight=1)

        prompt_label = ctk.CTkLabel(left_frame, text="Enter a prompt to generate a Quest, NPC, or Location:", font=ctk.CTkFont(weight="bold"))
        prompt_label.grid(row=0, column=0, padx=10, pady=(0,5), sticky="w")
        
        self.forge_input = ctk.CTkTextbox(left_frame, height=100)
        self.forge_input.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")
        self.forge_input.insert("0.0", "A grumpy dwarf blacksmith who has lost his lucky hammer.")
        
        self.forge_output = ctk.CTkTextbox(left_frame, state="disabled")
        self.forge_output.grid(row=2, column=0, padx=0, pady=5, sticky="nsew")

        button_frame = ctk.CTkFrame(left_frame)
        button_frame.grid(row=3, column=0, padx=0, pady=10, sticky="e")

        self.portrait_button = ctk.CTkButton(button_frame, text="Generate Portrait", command=self.start_portrait_thread, state="disabled")
        self.portrait_button.pack(side="right", padx=(5, 0))

        self.generate_button = ctk.CTkButton(button_frame, text="Generate", command=self.start_world_forge_thread)
        self.generate_button.pack(side="right", padx=(5, 0))

        save_button = ctk.CTkButton(button_frame, text="Save", command=self.save_world_forge_output)
        save_button.pack(side="right", padx=(0, 5))

        # --- Right Column: Image Display ---
        right_frame = ctk.CTkFrame(tab)
        right_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        self.portrait_label = ctk.CTkLabel(right_frame, text="NPC Portrait will appear here.", text_color="gray")
        self.portrait_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def save_world_forge_output(self):
        """Saves the generated NPC to the active campaign database."""
        if not self.current_campaign_path:
            print("No active campaign. Please create or open a campaign first.")
            # In a real app, you'd show a user-friendly dialog here.
            return

        description = self.forge_output.get("0.0", "end-1c")
        if not description.strip():
            print("No content to save.")
            return

        # Prompt user for NPC name
        dialog = CTkInputDialog(text="Enter a name for this NPC:", title="Save NPC")
        npc_name = dialog.get_input()

        if not npc_name or not npc_name.strip():
            print("Save cancelled: NPC name cannot be empty.")
            return

        try:
            conn = sqlite3.connect(self.current_campaign_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO npcs (name, description, portrait) VALUES (?, ?, ?)",
                (npc_name, description, self.portrait_image_data)
            )
            conn.commit()
            print(f"NPC '{npc_name}' saved to campaign.")
        except sqlite3.Error as e:
            print(f"Database error saving NPC: {e}")
        finally:
            if conn:
                conn.close()

    def setup_settings_tab(self):
        """Creates the widgets for the Settings tab."""
        tab = self.tab_view.tab("Settings")
        tab.grid_columnconfigure(0, weight=1)

        # --- API Key Management ---
        api_frame = ctk.CTkFrame(tab)
        api_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        api_frame.grid_columnconfigure(1, weight=1)

        api_label = ctk.CTkLabel(api_frame, text="OpenAI API Key:", font=ctk.CTkFont(weight="bold"))
        api_label.grid(row=0, column=0, padx=10, pady=10)

        self.api_key_entry = ctk.CTkEntry(api_frame, show="*")
        self.api_key_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Load existing key
        self.api_key_entry.insert(0, get_key(".env", "OPENAI_API_KEY") or "")

        save_api_button = ctk.CTkButton(api_frame, text="Save Key", command=self.save_api_key)
        save_api_button.grid(row=0, column=2, padx=10, pady=10)

        self.api_status_label = ctk.CTkLabel(api_frame, text="")
        self.api_status_label.grid(row=1, column=1, columnspan=2, padx=10, pady=(0, 10), sticky="w")

        # --- Microphone Settings ---
        mic_frame = ctk.CTkFrame(tab)
        mic_frame.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="ew")
        mic_frame.grid_columnconfigure(1, weight=1)

        mic_label = ctk.CTkLabel(mic_frame, text="Microphone:", font=ctk.CTkFont(weight="bold"))
        mic_label.grid(row=0, column=0, padx=10, pady=10)

        try:
            mic_names = sr.Microphone.list_microphone_names()
            if mic_names:
                self.mic_selection_var = ctk.StringVar(value=mic_names[self.selected_mic_index])
                mic_menu = ctk.CTkOptionMenu(mic_frame, variable=self.mic_selection_var, values=mic_names, command=self.on_mic_select)
                mic_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
            else:
                no_mic_label = ctk.CTkLabel(mic_frame, text="No microphones found.")
                no_mic_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
                self.selected_mic_index = None
        except Exception as e:
            error_mic_label = ctk.CTkLabel(mic_frame, text=f"Could not load microphones: {e}")
            error_mic_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
            self.selected_mic_index = None

    def on_mic_select(self, selected_mic_name):
        """Callback for when a microphone is selected from the dropdown."""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            self.selected_mic_index = mic_list.index(selected_mic_name)
            print(f"Microphone selected: {selected_mic_name} (index: {self.selected_mic_index})")
        except (ValueError, IndexError):
            print("Error selecting microphone.")
            self.selected_mic_index = None

    def save_api_key(self):
        """Saves the API key from the entry widget to the .env file."""
        new_key = self.api_key_entry.get()
        if new_key:
            set_key(".env", "OPENAI_API_KEY", new_key)
            self.api_status_label.configure(text="API Key saved. Please restart the application for changes to take effect.")
        else:
            self.api_status_label.configure(text="API Key cannot be empty.")

    def start_portrait_thread(self):
        """Starts the portrait generation in a new thread."""
        self.portrait_button.configure(state="disabled", text="Generating...")
        thread = threading.Thread(target=self.generate_npc_portrait)
        thread.daemon = True
        thread.start()

    def generate_npc_portrait(self):
        """Calls the DALL-E API to generate a portrait and prepares it for the UI."""
        npc_description = self.forge_output.get("0.0", "end-1c")
        if not npc_description.strip():
            print("No NPC description to generate a portrait from.")
            self.portrait_button.configure(state="normal", text="Generate Portrait")
            return

        # Truncate description to avoid exceeding API prompt limits
        max_desc_len = 3750
        if len(npc_description) > max_desc_len:
            npc_description = npc_description[:max_desc_len]

        prompt = (
            f"A digital painting of a fantasy character, focused on the face and upper body. "
            f"The character is: '{npc_description}'. "
            "Style: painterly, detailed, character concept art. No text, no signatures, no borders."
        )

        try:
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard"
            )
            image_url = response.data[0].url

            # Download the image and send it to the queue
            image_response = requests.get(image_url)
            image_response.raise_for_status() # Raise an exception for bad status codes
            image_data = image_response.content
            self.ambiance_queue.put(('portrait', image_data))

        except Exception as e:
            print(f"Error generating portrait: {e}")
            self.ambiance_queue.put(('log', f"ERROR: Could not generate portrait. {e}"))

        self.portrait_button.configure(state="normal", text="Generate Portrait")

    def analyze_text_for_audio(self, text):
        """Analyzes text for keywords and returns an audio trigger key."""
        text = text.lower()
        for trigger, keywords in AUDIO_TRIGGERS.items():
            if any(keyword in text for keyword in keywords):
                return trigger
        return None

    def trigger_audio(self, trigger_key):
        """Controls the audio playback based on a trigger key."""
        if trigger_key.startswith("sfx_"):
            # Play sound effects in a new, short-lived thread
            sfx_thread = threading.Thread(target=self.run_audio_player, args=(trigger_key,))
            sfx_thread.daemon = True
            sfx_thread.start()
        elif trigger_key.startswith("music_"):
            # If the requested music is already playing, do nothing
            if self.current_music_mood == trigger_key:
                return

            # Stop any currently playing music
            if self.audio_thread and self.audio_thread.is_alive():
                self.stop_audio_flag.set()
                self.audio_thread.join() # Wait for the thread to finish

            # Start new music
            self.current_music_mood = trigger_key
            self.ambiance_queue.put(('mood', trigger_key.replace("music_", "").capitalize()))
            self.stop_audio_flag.clear()
            self.audio_thread = threading.Thread(target=self.run_audio_player, args=(trigger_key,))
            self.audio_thread.daemon = True
            self.audio_thread.start()

    def run_audio_player(self, trigger_key):
        """
        Plays an audio file based on a trigger.
        Music loops until the stop flag is set. SFX play once.
        """
        filepath = AUDIO_FILES.get(trigger_key)
        if not filepath or not os.path.exists(filepath):
            print(f"Audio file not found for trigger: {trigger_key}")
            return

        try:
            song = AudioSegment.from_mp3(filepath)
            samples = np.array(song.get_array_of_samples()).reshape(-1, song.channels)

            if trigger_key.startswith("music_"):
                # Loop music until flag is set
                while not self.stop_audio_flag.is_set():
                    sd.play(samples, song.frame_rate)
                    sd.wait()
            else: # Sound effect
                sd.play(samples, song.frame_rate)
                sd.wait()

        except Exception as e:
            print(f"Error playing audio for {trigger_key}: {e}")

    def setup_rules_lawyer_tab(self):
        """Creates the widgets for the AI Rules Lawyer tab."""
        tab = self.tab_view.tab("AI Rules Lawyer")
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.rules_output = ctk.CTkTextbox(tab, state="disabled")
        self.rules_output.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.rules_input = ctk.CTkEntry(tab, placeholder_text="Ask a rule question...")
        self.rules_input.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.rules_input.bind("<Return>", self.start_rules_lawyer_thread)

        ask_button = ctk.CTkButton(tab, text="Ask", command=self.start_rules_lawyer_thread)
        ask_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

    def setup_encounter_architect_tab(self):
        """Creates the widgets for the Encounter Architect tab."""
        tab = self.tab_view.tab("Encounter Architect")

        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=2)
        
        party_frame = ctk.CTkFrame(tab)
        party_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        party_label = ctk.CTkLabel(party_frame, text="Party Setup", font=ctk.CTkFont(size=16, weight="bold"))
        party_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        level_label = ctk.CTkLabel(party_frame, text="Player Level:")
        level_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.party_level_var = ctk.StringVar(value="1")
        level_menu = ctk.CTkOptionMenu(party_frame, variable=self.party_level_var, values=[str(i) for i in range(1, 21)])
        level_menu.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        num_label = ctk.CTkLabel(party_frame, text="Number of Players:")
        num_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.party_size_var = ctk.StringVar(value="4")
        num_menu = ctk.CTkOptionMenu(party_frame, variable=self.party_size_var, values=[str(i) for i in range(1, 9)])
        num_menu.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        monster_frame = ctk.CTkFrame(tab)
        monster_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        monster_label = ctk.CTkLabel(monster_frame, text="Select Monsters", font=ctk.CTkFont(size=16, weight="bold"))
        monster_label.pack(padx=10, pady=10)

        monster_scroll_frame = ctk.CTkScrollableFrame(monster_frame)
        monster_scroll_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.monster_vars = {}
        for monster_name in sorted(self.monsters.keys()):
            self.monster_vars[monster_name] = ctk.IntVar()
            cb = ctk.CTkCheckBox(monster_scroll_frame, text=monster_name, variable=self.monster_vars[monster_name])
            cb.pack(anchor="w", padx=10)

        calc_frame = ctk.CTkFrame(tab)
        calc_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        calc_button = ctk.CTkButton(calc_frame, text="Calculate Difficulty", command=self.calculate_encounter)
        calc_button.pack(side="left", padx=10, pady=10)

        self.suggestion_button = ctk.CTkButton(calc_frame, text="Get AI Suggestions", command=self.start_suggestion_thread, state="disabled")
        self.suggestion_button.pack(side="left", padx=10, pady=10)

        self.result_label = ctk.CTkLabel(calc_frame, text="Result: -", font=ctk.CTkFont(size=14))
        self.result_label.pack(side="left", padx=10, pady=10)

        # --- Suggestion Frame ---
        suggestion_frame = ctk.CTkFrame(tab)
        suggestion_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        suggestion_frame.grid_columnconfigure(0, weight=1)
        suggestion_frame.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1) # Allow this row to expand

        self.suggestion_output = ctk.CTkTextbox(suggestion_frame, state="disabled", wrap="word")
        self.suggestion_output.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def update_textbox(self, textbox, text):
        """Helper function to safely update a textbox from any thread."""
        textbox.configure(state="normal")
        textbox.delete("0.0", "end")
        textbox.insert("0.0", text)
        textbox.configure(state="disabled")

    def append_to_textbox(self, textbox, text):
        """Helper function to append text to a textbox."""
        textbox.configure(state="normal")
        textbox.insert("end", text)
        textbox.configure(state="disabled")

    def start_world_forge_thread(self):
        """Starts a new thread for the world forge to prevent UI freezing."""
        self.generate_button.configure(state="disabled", text="Generating...")
        thread = threading.Thread(target=self.run_world_forge)
        thread.start()

    def run_world_forge(self):
        """The actual AI call for the World Forge."""
        user_prompt = self.forge_input.get("0.0", "end").strip()
        system_prompt = (
            "You are a creative assistant for a Dungeon Master. "
            "Generate detailed, engaging content for a TTRPG campaign. Use evocative language."
        )
        full_prompt = f"{system_prompt}\n\nUser Request: '{user_prompt}'"
        
        response = self.world_forge_llm.invoke(full_prompt)
        
        self.update_textbox(self.forge_output, response.content)
        self.generate_button.configure(state="normal", text="Generate")
        self.portrait_button.configure(state="normal")

    def start_rules_lawyer_thread(self, event=None):
        """Starts a new thread for the rules lawyer."""
        question = self.rules_input.get().strip()
        if not question:
            return
            
        self.append_to_textbox(self.rules_output, f"You: {question}\n\n")
        self.rules_input.delete(0, "end")
        
        self.tab_view.tab("AI Rules Lawyer").winfo_children()[2].configure(state="disabled")
        
        thread = threading.Thread(target=self.run_rules_lawyer, args=(question,))
        thread.start()

    def run_rules_lawyer(self, question):
        """The actual AI call for the Rules Lawyer."""
        result = self.rules_lawyer_chain({
            "question": question, 
            "chat_history": self.rules_chat_history
        })
        answer = result['answer']
        
        self.append_to_textbox(self.rules_output, f"AI: {answer}\n-----------------\n\n")
        
        self.rules_chat_history.append((question, answer))
        
        self.tab_view.tab("AI Rules Lawyer").winfo_children()[2].configure(state="normal")

    def start_suggestion_thread(self):
        """Starts the encounter suggestion in a new thread."""
        self.suggestion_button.configure(state="disabled", text="Getting suggestion...")
        thread = threading.Thread(target=self.get_encounter_suggestions)
        thread.daemon = True
        thread.start()

    def get_encounter_suggestions(self):
        """Gathers context, calls the LLM, and puts the suggestion in the queue."""
        try:
            # 1. Gather context
            party_level = self.party_level_var.get()
            party_size = self.party_size_var.get()
            selected_monsters = [name for name, var in self.monster_vars.items() if var.get() == 1]
            monster_list_str = ", ".join(selected_monsters)
            difficulty = self.result_label.cget("text") # Get the text from the result label

            # 2. Construct the prompt
            prompt = (
                f"You are an expert Dungeon Master providing advice on a Dungeons & Dragons encounter. "
                f"The party consists of {party_size} level {party_level} adventurers.\n"
                f"The current encounter design includes the following monsters: {monster_list_str}.\n"
                f"My calculation shows this encounter's difficulty is '{difficulty}'.\n\n"
                "Please provide a brief, actionable suggestion to make this encounter more creative, thematic, or mechanically interesting. "
                "Do not just suggest adding more monsters to increase difficulty. "
                "Focus on synergy, environment, or a simple, unique monster ability. For example: 'Consider having the goblins use nets to restrain players before the bugbear attacks.' or 'This fight could be more interesting if it happened on a rickety rope bridge.'"
            )

            # 3. Call the LLM
            response = self.world_forge_llm.invoke(prompt) # Re-using the gpt-4o instance
            suggestion = response.content

            # 4. Put the result in the queue
            self.ambiance_queue.put(('suggestion', suggestion))

        except Exception as e:
            print(f"Error getting AI suggestion: {e}")
            self.ambiance_queue.put(('suggestion', f"An error occurred: {e}"))

    def calculate_encounter(self):
        """Calculates and displays the encounter difficulty."""
        party_level = int(self.party_level_var.get())
        party_size = int(self.party_size_var.get())
        
        selected_monsters = [name for name, var in self.monster_vars.items() if var.get() == 1]
        
        if not selected_monsters:
            self.result_label.configure(text="Result: Please select at least one monster.")
            self.suggestion_button.configure(state="disabled")
            return

        total_xp = sum(self.monsters[name]["XP"] for name in selected_monsters)

        num_monsters = len(selected_monsters)
        multiplier = ENCOUNTER_MULTIPLIERS.get(num_monsters, 4.0)
        
        adjusted_xp = total_xp * multiplier

        thresholds = XP_THRESHOLDS.get(party_level, XP_THRESHOLDS[5])
        party_thresholds = {
            "easy": thresholds["easy"] * party_size,
            "medium": thresholds["medium"] * party_size,
            "hard": thresholds["hard"] * party_size,
            "deadly": thresholds["deadly"] * party_size
        }

        difficulty = "Trivial"
        if adjusted_xp >= party_thresholds["deadly"]:
            difficulty = "Deadly"
        elif adjusted_xp >= party_thresholds["hard"]:
            difficulty = "Hard"
        elif adjusted_xp >= party_thresholds["medium"]:
            difficulty = "Medium"
        elif adjusted_xp >= party_thresholds["easy"]:
            difficulty = "Easy"

        result_text = f"Result: {difficulty} ({int(adjusted_xp)} adjusted XP)"
        self.result_label.configure(text=result_text)
        self.suggestion_button.configure(state="normal") # Enable the button

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    rules_chain = initialize_ai_rules_lawyer()
    forge_llm = initialize_world_forge()
    
    app = DMCommandCenterApp(rules_lawyer_chain=rules_chain, world_forge_llm=forge_llm)
    app.mainloop()