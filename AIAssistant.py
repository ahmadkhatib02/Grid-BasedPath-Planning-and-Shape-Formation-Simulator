import tkinter as tk
from tkinter import scrolledtext, Frame, Button, Entry
import random

class AIAssistant:
    """AI Assistant class that provides help and explanations to users."""

    def __init__(self, parent):
        self.parent = parent

        self.window = tk.Toplevel(parent)
        self.window.title("AI Assistant")
        self.window.geometry("400x500")
        self.window.minsize(300, 400)
        self.window.configure(bg="#F8F9FA")

        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        self.window.withdraw()

        self.create_chat_interface()

        self.greeting_messages = [
            "Hello! I'm your AI assistant. How can I help you today?",
            "Welcome! I'm here to help you with the Interactive Grid application.",
            "Hi there! Need help with the application? Just ask me!",
            "Greetings! I can help you understand the algorithms and navigate the UI."
        ]
        self.navigation_help = {
            "general": "This application allows you to simulate agents forming shapes using different pathfinding algorithms. The left side shows the grid, and the right side contains controls.",
            "grid_size": "You can change the grid size by entering a number and clicking 'Apply'. Larger grids will have smaller cells.",
            "agent_count": "Control the number of agents by entering a value and clicking 'Apply'. More agents can form shapes faster but may cause congestion.",
            "speed": "Adjust the movement speed using the slider. Moving it to the left makes agents move faster.",
            "shapes": "Choose from predefined shapes (Rectangle, Triangle, Circle) or create a custom shape by clicking cells on the grid.",
            "algorithms": "Select different pathfinding algorithms from the dropdown menu to see how they perform differently.",
            "obstacles": "Add random obstacles or clear them using the buttons. You can also click on cells to add/remove obstacles manually.",
            "actions": "Click 'Do the Shape' to start the simulation and 'Reset' to clear the grid and start over.",
            "metrics": "The Performance Metrics section shows statistics about the current algorithm's performance."
        }
        self.algorithm_explanations = {
            "A*": "A* (A-Star) is an informed search algorithm that uses a heuristic to estimate the cost to reach the goal. "
                 "In this application, A* finds the shortest path from each agent to its target while avoiding obstacles. "
                 "It's generally the most efficient algorithm for pathfinding, balancing speed and path optimality.",

            "BFS": "Breadth-First Search explores all neighboring cells before moving to the next level of cells. "
                  "In this application, BFS guarantees the shortest path in terms of the number of steps, but it explores "
                  "more cells than necessary since it searches in all directions equally.",

            "DFS": "Depth-First Search explores as far as possible along each branch before backtracking. "
                  "In this application, DFS may not find the shortest path, but it can be faster in some maze-like "
                  "environments. It uses less memory than BFS but can get stuck in long paths.",

            "Minimax": "Minimax is a decision-making algorithm that minimizes the possible loss for a worst-case scenario. "
                      "In this application, it's adapted for pathfinding by treating obstacles and other agents as 'opponents'. "
                      "It's more computationally intensive but can handle adversarial situations.",

            "Alpha-Beta": "Alpha-Beta Pruning is an optimization of the Minimax algorithm that reduces the number of nodes evaluated. "
                         "In this application, it works like Minimax but is more efficient by skipping evaluations of moves that "
                         "won't affect the final decision. The metrics show how many nodes were pruned.",

            "Expectimax": "Expectimax is a variation of Minimax that handles probabilistic outcomes. "
                         "In this application, it models uncertainty in agent movements, making it useful when "
                         "other agents' behaviors are not deterministic. It's especially useful in crowded scenarios."
        }

    def create_chat_interface(self):
        """Create the chat interface components."""
        history_frame = Frame(self.window, bg="#F8F9FA")
        history_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.chat_history = scrolledtext.ScrolledText(
            history_frame,
            wrap=tk.WORD,
            bg="#FFFFFF",
            font=("Arial", 10),
            state=tk.DISABLED
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)

        input_frame = Frame(self.window, bg="#F8F9FA")
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.user_input = Entry(
            input_frame,
            font=("Arial", 10),
            bg="#FFFFFF"
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.process_input)

        send_button = Button(
            input_frame,
            text="Send",
            command=self.process_input,
            bg="#3498DB",
            fg="white",
            activebackground="#2980B9"
        )
        send_button.pack(side=tk.RIGHT)

        quick_help_frame = Frame(self.window, bg="#F8F9FA")
        quick_help_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        help_topics = ["UI Help", "Algorithms", "Shapes", "Controls"]

        for topic in help_topics:
            button = Button(
                quick_help_frame,
                text=topic,
                command=lambda t=topic: self.show_quick_help(t),
                bg="#95A5A6",
                fg="white",
                activebackground="#7F8C8D",
                padx=5
            )
            button.pack(side=tk.LEFT, padx=2)

    def show(self):
        """Show the assistant window."""
        self.window.deiconify()
        self.greet()

    def hide(self):
        """Hide the assistant window."""
        self.window.withdraw()

    def greet(self):
        """Display a random greeting message."""
        greeting = random.choice(self.greeting_messages)
        self.display_message(greeting, "assistant")

    def display_message(self, message, sender):
        """Display a message in the chat history."""
        self.chat_history.config(state=tk.NORMAL)

        if sender == "user":
            self.chat_history.insert(tk.END, "You: ", "user_tag")
            self.chat_history.tag_configure("user_tag", foreground="#2980B9", font=("Arial", 10, "bold"))
        else:
            self.chat_history.insert(tk.END, "Assistant: ", "assistant_tag")
            self.chat_history.tag_configure("assistant_tag", foreground="#27AE60", font=("Arial", 10, "bold"))

        self.chat_history.insert(tk.END, f"{message}\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def process_input(self, event=None):
        """Process user input and generate a response."""
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        self.display_message(user_text, "user")

        self.user_input.delete(0, tk.END)

        response = self.generate_response(user_text)
        self.display_message(response, "assistant")

    def generate_response(self, user_input):
        """Generate a response based on user input."""
        user_input = user_input.lower()

        for algo, explanation in self.algorithm_explanations.items():
            if algo.lower() in user_input:
                return explanation

        for topic, help_text in self.navigation_help.items():
            if topic in user_input:
                return help_text

        if any(word in user_input for word in ["hello", "hi", "hey", "greetings"]):
            return random.choice(self.greeting_messages)

        if any(word in user_input for word in ["help", "assist", "guide"]):
            return "I can help you with navigating the UI, understanding algorithms, or explaining features. What would you like to know about?"

        if any(word in user_input for word in ["algorithm", "pathfinding", "search"]):
            return "I can explain several algorithms: A*, BFS, DFS, Minimax, Alpha-Beta, and Expectimax. Which one would you like to learn about?"

        if any(word in user_input for word in ["ui", "interface", "controls", "how to"]):
            return self.navigation_help["general"]

        return "I'm not sure I understand. You can ask me about specific algorithms, UI navigation, or how to use different features of the application."

    def show_quick_help(self, topic):
        """Show quick help based on the selected topic."""
        if topic == "UI Help":
            self.display_message("What part of the UI would you like help with? (grid_size, agent_count, speed, shapes, obstacles, actions, metrics)", "assistant")

        elif topic == "Algorithms":
            self.display_message("I can explain these algorithms: A*, BFS, DFS, Minimax, Alpha-Beta, and Expectimax. Which one interests you?", "assistant")

        elif topic == "Shapes":
            self.display_message(
                "The application supports several shapes:\n"
                "- Rectangle: A simple rectangular shape\n"
                "- Triangle: A triangular shape\n"
                "- Circle: A circular/oval shape\n"
                "- Custom: Create your own shape by clicking on the grid",
                "assistant"
            )

        elif topic == "Controls":
            self.display_message(
                "Main controls:\n"
                "- Grid Size: Change the dimensions of the grid\n"
                "- Agent Count: Set how many agents to use\n"
                "- Movement Speed: Control how fast agents move\n"
                "- Algorithm: Select which pathfinding algorithm to use\n"
                "- Do the Shape: Start the simulation\n"
                "- Reset: Clear the grid and start over",
                "assistant"
            )
