import os
from PIL import Image
from google import genai
from google.genai import types

class GeminiVisionChat:
    def __init__(self, api_key: str = None):
        # 1. Client initialisieren (zieht den Key automatisch aus os.environ["GEMINI_API_KEY"])
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        
        # 2. Modell wählen (gemini-2.5-flash ist extrem schnell, präzise und multimodal)
        self.model_id = "gemini-2.5-flash"
        
        # 3. Nativer Chat-Session-Speicher von Google
        self.chat = self.client.chats.create(model=self.model_id)

    def send_message(self, user_text: str, image_path: str = None):
        """Sendet Text und optional ein Bild nativ an Gemini"""
        contents = []

        # Falls ein Bild mitgegeben wird, laden wir es einfach mit PIL
        if image_path:
            if not os.path.exists(image_path):
                return f"Fehler: Bild unter '{image_path}' nicht gefunden."
            img = Image.open(image_path)
            contents.append(img)

        # Text hinzufügen
        contents.append(user_text)

        try:
            # Senden an die native Chat-Session
            response = self.chat.send_message(contents)
            return response.text
        except Exception as e:
            return f"Fehler bei der Anfrage: {str(e)}"

    def reset_history(self):
        """Setzt die Unterhaltung zurück"""
        self.chat = self.client.chats.create(model=self.model_id)
        return "Chat-Historie zurückgesetzt!"


# ==================== INTERAKTIVE SCHLEIFE ====================
def interactive_chat():
    # API-Key Sicherstellung
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = input("Bitte deinen Gemini API Key eingeben: ").strip()

    bot = GeminiVisionChat()

    print("\n" + "="*50)
    print("Nativer Gemini Vision Chat gestartet!")
    print("Befehle: 'exit', 'reset', '/image <pfad>'")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nDu: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Chat beendet.")
                break

            if user_input.lower() == "reset":
                print(bot.reset_history())
                continue

            image_path = None
            if user_input.startswith("/image "):
                parts = user_input.split(" ", 1)
                if len(parts) > 1:
                    image_path = parts[1].strip()
                    user_input = input("Deine Frage zum Bild: ").strip()

            print("Gemini denkt nach...", end="", flush=True)
            antwort = bot.send_message(user_text=user_input, image_path=image_path)
            print("\r" + " "*25 + "\r", end="") # Zeile leeren
            
            print(f"\nGemini: {antwort}")

        except KeyboardInterrupt:
            print("\nAbgebrochen.")
            break

if __name__ == "__main__":
    interactive_chat()
