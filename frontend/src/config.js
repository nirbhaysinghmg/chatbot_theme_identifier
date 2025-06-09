// src/config.js
const config = {
  companyName: "Healthcare AI",
  companyLogo: "/assets/images/apollo-tyres-logo-png_seeklogo-314374.png",
  agentName: "Healthcare AI Assistant",
  projectName: "Healthcare AI",
  chatUrl: "ws://localhost:8008/ws", // Changed from ws://localhost:8000/ws/chat
  phoneSubmitUrl: "http://localhost:8008/api/mobile",
  theme: {
    primaryColor: "#0066cc",
    secondaryColor: "#f0f0f0",
    backgroundColor: "#ffffff",
    textColor: "#333333",
  },
  // Customizable introductory message
  introductionText: `
### 👋 Welcome to our AI Help Chat.
  `,
  inputPlaceholder: "Type your question here...",
};

export default config;
