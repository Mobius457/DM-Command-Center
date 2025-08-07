import os
import csv
import customtkinter as ctk
import threading
import queue
import speech_recognition as sr
from dotenv import load_dotenv

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


        # --- AI Model & Data Storage ---
        self.rules_lawyer_chain = rules_lawyer_chain
        self.world_forge_llm = world_forge_llm
        self.rules_chat_history = []
        self.monsters = self.load_monsters()

        # --- Ambiance Engine State ---
        self.is_listening = False
        self.listener_thread = None
        self.ambiance_queue = queue.Queue()

        # --- Window Configuration ---
        self.title("AI Dungeon Master's Command Center")
        self.geometry("800x650")
        ctk.set_appearance_mode("dark")

        # --- Create Tabs ---
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(padx=20, pady=20, fill="both", expand=True)
        self.tab_view.add("World Forge")
        self.tab_view.add("AI Rules Lawyer")
        self.tab_view.add("Encounter Architect")
        self.tab_view.add("Ambiance Engine")
        
        # --- Configure Tabs ---
        self.setup_world_forge_tab()
        self.setup_rules_lawyer_tab()
        self.setup_encounter_architect_tab()
        self.setup_ambiance_tab()

        # --- Start Queue Processor ---
        self.process_ambiance_queue()

    def process_ambiance_queue(self):
        """Processes messages from the ambiance queue to update the UI safely."""
        try:
            msg_type, data = self.ambiance_queue.get_nowait()
            if msg_type == 'log':
                self.append_to_textbox(self.transcription_log, data + "\n")
            elif msg_type == 'status':
                self.ambiance_status_label.configure(text=f"Status: {data}")
            elif msg_type == 'button_state':
                self.ambiance_button.configure(state=data)
            elif msg_type == 'button_text':
                self.ambiance_button.configure(text=data)
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
        mic = sr.Microphone()

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

        # Log Frame
        log_frame = ctk.CTkFrame(tab)
        log_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        log_label = ctk.CTkLabel(log_frame, text="Transcription Log")
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
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        prompt_label = ctk.CTkLabel(tab, text="Enter a prompt to generate a Quest, NPC, or Location:")
        prompt_label.grid(row=0, column=0, padx=10, pady=(10,0), sticky="w")
        
        self.forge_input = ctk.CTkTextbox(tab, height=100)
        self.forge_input.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.forge_input.insert("0.0", "A grumpy dwarf blacksmith who has lost his lucky hammer.")
        
        self.forge_output = ctk.CTkTextbox(tab, state="disabled")
        self.forge_output.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        
        generate_button = ctk.CTkButton(tab, text="Generate", command=self.start_world_forge_thread)
        generate_button.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=3)

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

        self.result_label = ctk.CTkLabel(calc_frame, text="Result: -", font=ctk.CTkFont(size=14))
        self.result_label.pack(side="left", padx=10, pady=10)

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
        self.tab_view.tab("World Forge").winfo_children()[3].configure(state="disabled", text="Generating...")
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
        self.tab_view.tab("World Forge").winfo_children()[3].configure(state="normal", text="Generate")

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

    def calculate_encounter(self):
        """Calculates and displays the encounter difficulty."""
        party_level = int(self.party_level_var.get())
        party_size = int(self.party_size_var.get())
        
        selected_monsters = [name for name, var in self.monster_vars.items() if var.get() == 1]
        
        if not selected_monsters:
            self.result_label.configure(text="Result: Please select at least one monster.")
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

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    rules_chain = initialize_ai_rules_lawyer()
    forge_llm = initialize_world_forge()
    
    app = DMCommandCenterApp(rules_lawyer_chain=rules_chain, world_forge_llm=forge_llm)
    app.mainloop()