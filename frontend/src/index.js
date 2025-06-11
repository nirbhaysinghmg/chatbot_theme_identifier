import ReactDOM from "react-dom/client";
import ChatWidget from "./components/ChatWidget";
import config from "./config";
import "./components/ChatWidget.css";

// Development configuration
const devConfig = {
  ...config,
  // Override any config for development
  companyName: "Healthcare AI",
  showButton: true,
  showGreeting: false,
};

// Define global chatbot functions first, before any DOM operations
window.toggleChatbot = () => {
  const chatbot = document.getElementById("chatbot");
  const button = document.getElementById("healthcare-ai-button");

  if (chatbot) {
    if (chatbot.classList.contains("hidden")) {
      // Open chatbot
      chatbot.style.display = "block";
      requestAnimationFrame(() => {
        chatbot.classList.remove("hidden");
      });
      if (button) button.style.display = "none";
    } else {
      // Close chatbot
      chatbot.classList.add("hidden");
      setTimeout(() => {
        chatbot.style.display = "none";
      }, 300);
      if (button) button.style.display = "flex";
    }
  }
};

window.closeChatbot = () => {
  const chatbot = document.getElementById("chatbot");
  const button = document.getElementById("healthcare-ai-button");

  if (chatbot) {
    chatbot.classList.add("hidden");
    setTimeout(() => {
      chatbot.style.display = "none";
    }, 300);
    if (button) button.style.display = "flex";
  }
};

window.openChatbot = () => {
  const chatbot = document.getElementById("chatbot");
  const button = document.getElementById("healthcare-ai-button");

  if (chatbot) {
    chatbot.style.display = "block";
    requestAnimationFrame(() => {
      chatbot.classList.remove("hidden");
    });
    if (button) button.style.display = "none";
  }
};

// Initialize function to set up the chatbot
const initChatbot = () => {
  console.log("Initializing Healthcare AI Chatbot...");

  // Create the chatbot container if it doesn't exist
  let container = document.getElementById("healthcare-ai-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "healthcare-ai-container";
    document.body.appendChild(container);
  }

  // Create the chat button
  createChatButton();

  // Create the chatbot widget
  createChatWidget();

  console.log("Healthcare AI Chatbot initialized in development mode");
};

// Create the chat button
const createChatButton = () => {
  const buttonId = "healthcare-ai-button";
  let button = document.getElementById(buttonId);

  if (!button) {
    button = document.createElement("div");
    button.id = buttonId;
    button.className = "healthcare-ai-button";
    button.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
    `;
    document.body.appendChild(button);

    // Style the button
    Object.assign(button.style, {
      position: "fixed",
      bottom: "20px",
      right: "20px",
      width: "60px",
      height: "60px",
      borderRadius: "50%",
      backgroundColor: devConfig.theme?.primaryColor || "#0066cc",
      color: "white",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      cursor: "pointer",
      boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
      zIndex: "999",
      transition: "all 0.3s ease",
    });

    // Add hover effect
    button.addEventListener("mouseenter", () => {
      button.style.transform = "scale(1.05)";
      button.style.boxShadow = "0 6px 12px rgba(0,0,0,0.3)";
    });

    button.addEventListener("mouseleave", () => {
      button.style.transform = "scale(1)";
      button.style.boxShadow = "0 4px 8px rgba(0,0,0,0.2)";
    });

    // Add click event to open chatbot
    button.addEventListener("click", window.openChatbot);
  }

  return button;
};

// Create the chatbot widget
const createChatWidget = () => {
  let chatbotContainer = document.getElementById("chatbot");

  if (!chatbotContainer) {
    chatbotContainer = document.createElement("div");
    chatbotContainer.id = "chatbot";
    chatbotContainer.className = "hidden";
    chatbotContainer.style.display = "none";
    document.body.appendChild(chatbotContainer);
  }

  // Render the ChatWidget
  const root = ReactDOM.createRoot(chatbotContainer);
  root.render(<ChatWidget config={devConfig} />);
};

// Initialize the chatbot when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initChatbot);
} else {
  initChatbot();
}
