// ChatWidget.jsx

import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useChatSocket } from "../hooks/useChatSocket";
import defaultConfig from "../config";
import "./ChatWidget.css"; // Import CSS from the same directory
import DocumentViewer from "./DocumentViewer";

const FeedbackPrompt = ({ onYes, onNo }) => (
  <div
    style={{
      marginTop: 8,
      marginBottom: 8,
      background: "#f7f7f7",
      borderRadius: 8,
      padding: 12,
      display: "flex",
      alignItems: "center",
      gap: 12,
    }}
  >
    <span>Was this helpful?</span>
    <button
      style={{
        background: "#4caf50",
        color: "#fff",
        border: "none",
        borderRadius: 4,
        padding: "4px 12px",
        cursor: "pointer",
      }}
      onClick={onYes}
    >
      Yes
    </button>
    <button
      style={{
        background: "#f44336",
        color: "#fff",
        border: "none",
        borderRadius: 4,
        padding: "4px 12px",
        cursor: "pointer",
      }}
      onClick={onNo}
    >
      No
    </button>
  </div>
);

const ThankYouMessage = () => (
  <div
    style={{
      marginTop: 8,
      marginBottom: 8,
      background: "#e8f5e9",
      color: "#2e7d32",
      borderRadius: 8,
      padding: 12,
      display: "flex",
      alignItems: "center",
      gap: 12,
      animation: "fadeOut 2s ease-in-out forwards",
    }}
  >
    <span>Thank you for your feedback!</span>
  </div>
);

const FeedbackForm = ({ onClose, onSubmit }) => {
  const [issues, setIssues] = useState([]);
  const [otherText, setOtherText] = useState("");
  const [supportOption, setSupportOption] = useState("");
  const [execMethod, setExecMethod] = useState("");

  const issueOptions = [
    { value: "off-topic", label: "The answer was off-topic" },
    { value: "too-short", label: "It was too short" },
    { value: "too-complex", label: "It was too complex" },
    { value: "different-help", label: "I need a different kind of help" },
    { value: "other", label: "Other:" },
  ];

  const handleIssueChange = (val) => {
    if (issues.includes(val)) {
      setIssues(issues.filter((i) => i !== val));
      if (val === "other") setOtherText("");
    } else {
      setIssues([...issues, val]);
    }
  };

  const canSubmit =
    issues.length > 0 &&
    supportOption &&
    (supportOption !== "talk-exec" || execMethod);

  const handleSubmit = () => {
    const feedback = {
      issues,
      otherText: issues.includes("other") ? otherText : "",
      supportOption,
      execMethod: supportOption === "talk-exec" ? execMethod : "",
    };
    onSubmit(feedback);
  };

  return (
    <div
      style={{
        marginTop: 8,
        marginBottom: 8,
        background: "#fffbe6",
        border: "1px solid #ffe58f",
        borderRadius: 8,
        padding: 16,
        maxWidth: 400,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>
        What seems to be the issue?{" "}
        <span style={{ fontWeight: 400, fontSize: 13 }}>
          (Choose one or more)
        </span>
      </div>
      <div style={{ marginBottom: 12 }}>
        {issueOptions.map((opt) => (
          <div key={opt.value} style={{ marginBottom: 4 }}>
            <label>
              <input
                type="checkbox"
                checked={issues.includes(opt.value)}
                onChange={() => handleIssueChange(opt.value)}
                style={{ marginRight: 6 }}
              />
              {opt.label}
              {opt.value === "other" && issues.includes("other") && (
                <input
                  type="text"
                  value={otherText}
                  onChange={(e) => setOtherText(e.target.value)}
                  placeholder="Please specify"
                  style={{
                    marginLeft: 8,
                    padding: 2,
                    borderRadius: 4,
                    border: "1px solid #ccc",
                    width: 140,
                  }}
                />
              )}
            </label>
          </div>
        ))}
      </div>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>Would you like to:</div>
      <div style={{ marginBottom: 12 }}>
        <label>
          <input
            type="radio"
            name="support"
            value="rephrase"
            checked={supportOption === "rephrase"}
            onChange={() => setSupportOption("rephrase")}
            style={{ marginRight: 6 }}
          />
          Try rephrasing your question
        </label>
        <br />
        <label>
          <input
            type="radio"
            name="support"
            value="talk-exec"
            checked={supportOption === "talk-exec"}
            onChange={() => setSupportOption("talk-exec")}
            style={{ marginRight: 6 }}
          />
          Talk to a human executive
        </label>
      </div>
      {supportOption === "talk-exec" && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            Preferred method to connect?
          </div>
          <label>
            <input
              type="radio"
              name="exec-method"
              value="chat"
              checked={execMethod === "chat"}
              onChange={() => setExecMethod("chat")}
              style={{ marginRight: 6 }}
            />
            Chat now
          </label>
          <br />
          <label>
            <input
              type="radio"
              name="exec-method"
              value="call"
              checked={execMethod === "call"}
              onChange={() => setExecMethod("call")}
              style={{ marginRight: 6 }}
            />
            Schedule a call
          </label>
          <br />
          <label>
            <input
              type="radio"
              name="exec-method"
              value="email"
              checked={execMethod === "email"}
              onChange={() => setExecMethod("email")}
              style={{ marginRight: 6 }}
            />
            Email me back
          </label>
        </div>
      )}
      <div style={{ display: "flex", gap: 8 }}>
        <button
          style={{
            background: canSubmit ? "#0066cc" : "#ccc",
            color: "#fff",
            border: "none",
            borderRadius: 4,
            padding: "4px 12px",
            cursor: canSubmit ? "pointer" : "not-allowed",
          }}
          onClick={handleSubmit}
          disabled={!canSubmit}
        >
          Submit
        </button>
        <button
          style={{
            background: "#eee",
            color: "#333",
            border: "none",
            borderRadius: 4,
            padding: "4px 12px",
            cursor: "pointer",
          }}
          onClick={onClose}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

const styles = `
  @keyframes fadeOut {
    0% {
      opacity: 1;
    }
    70% {
      opacity: 1;
    }
    100% {
      opacity: 0;
    }
  }
`;

const styleSheet = document.createElement("style");
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);

// Add new components for document viewing and theme analysis
const ThemeAnalysis = ({ themes }) => {
  return (
    <div className="theme-analysis">
      <h3>Theme Analysis</h3>
      {themes.map((theme, index) => (
        <div key={index} className="theme-block">
          <h4>
            Theme {index + 1}: {theme.title}
          </h4>
          <p>{theme.summary}</p>
          <div className="theme-citations">
            <h5>Supporting Documents:</h5>
            <ul>
              {theme.citations.map((citation, idx) => (
                <li key={idx}>
                  {citation.doc_id} (Page {citation.page}, Lines{" "}
                  {citation.lines})
                </li>
              ))}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
};

const DocumentAnswers = ({ answers }) => {
  return (
    <div className="document-answers">
      <h3>Document Answers</h3>
      <table className="answers-table">
        <thead>
          <tr>
            <th>Document ID</th>
            <th>Answer</th>
            <th>Citation</th>
          </tr>
        </thead>
        <tbody>
          {answers.map((answer, index) => (
            <tr key={index}>
              <td>{answer.doc_id}</td>
              <td>{answer.answer}</td>
              <td>
                Page {answer.citation.page}, line {answer.citation.lines}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Add DocumentUploadModal component
const DocumentUploadModal = ({ isOpen, onClose, onUpload }) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    setError("");
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError("Please select at least one file");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append("file", file);
      });

      // Get session ID from localStorage
      const sessionId = localStorage.getItem("healthcare_session_id");
      console.log("Uploading with session ID:", sessionId);

      const response = await fetch(
        `https://chatbot-theme-identifier-kzpk.onrender.com/upload/document?session_id=${
          sessionId || ""
        }`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const result = await response.json();
      console.log("Upload result:", result);
      onUpload(result);
      onClose();
    } catch (err) {
      console.error("Upload error:", err);
      setError("Failed to upload files. Please try again.");
    } finally {
      setUploading(false);
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h3>Upload Documents</h3>
          <button onClick={onClose} className="close-button">
            ×
          </button>
        </div>
        <div className="modal-body">
          <div className="upload-area">
            <input
              type="file"
              multiple
              onChange={handleFileChange}
              ref={fileInputRef}
              accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png"
            />
            <p className="file-types">
              Supported formats: PDF, DOC, DOCX, TXT, JPG, JPEG, PNG
            </p>
          </div>
          {files.length > 0 && (
            <div className="selected-files">
              <h4>Selected Files:</h4>
              <ul>
                {files.map((file, index) => (
                  <li key={index}>{file.name}</li>
                ))}
              </ul>
            </div>
          )}
          {error && <div className="error-message">{error}</div>}
        </div>
        <div className="modal-footer">
          <button
            className="cancel-button"
            onClick={onClose}
            disabled={uploading}
          >
            Cancel
          </button>
          <button
            className="upload-button"
            onClick={handleUpload}
            disabled={uploading || files.length === 0}
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>
      </div>
    </div>
  );
};

const ChatWidget = ({ config: userConfig }) => {
  // Merge config with defaults
  const cfg = { ...defaultConfig, ...userConfig };

  // Chat state
  const [chatHistory, setChatHistory] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [fullScreen] = useState(true);

  // Scheduling form state
  const [showScheduleForm, setShowScheduleForm] = useState(false);
  const [scheduleFormData, setScheduleFormData] = useState({
    name: "",
    phoneNumber: "",
    vehicleType: "",
  });
  const [scheduleFormSubmitted, setScheduleFormSubmitted] = useState(false);
  const [scheduleError, setScheduleError] = useState("");

  // Track session start time
  const [sessionStartTime, setSessionStartTime] = useState(Date.now());
  useEffect(() => {
    setSessionStartTime(Date.now());
  }, []);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // WebSocket connection
  const { sendMessage, connectionStatus, trackUserAction } = useChatSocket(
    setChatHistory,
    setStreaming,
    cfg.chatUrl
  );

  // Add session tracking on component mount
  useEffect(() => {
    // Clear all sessions when component mounts
    const clearSessions = async () => {
      try {
        await fetch(
          "https://chatbot-theme-identifier-kzpk.onrender.com/sessions/clear",
          {
            method: "DELETE",
          }
        );
        // Clear local storage
        localStorage.removeItem("healthcare_session_id");
      } catch (error) {
        console.error("Error clearing sessions:", error);
      }
    };
    clearSessions();

    // Track page load/refresh as a new session
    trackUserAction("session_start", {
      referrer: document.referrer,
      userAgent: navigator.userAgent,
    });
  }, [trackUserAction]);

  // Seed the initial system message
  useEffect(() => {
    setChatHistory([{ role: "system", text: cfg.introductionText }]);
  }, [cfg.introductionText]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, showScheduleForm]);

  // Auto-resize textarea - modified to respect fixed height
  useEffect(() => {
    if (textareaRef.current) {
      // Only adjust height if content exceeds the fixed height
      const scrollHeight = textareaRef.current.scrollHeight;
      const fixedHeight = 55; // Match the CSS height

      if (scrollHeight > fixedHeight) {
        // Allow content to scroll within the fixed height
        textareaRef.current.style.overflowY = "auto";
      } else {
        // Hide scrollbar when not needed
        textareaRef.current.style.overflowY = "hidden";
      }
    }
  }, [input]);

  // Add new state for documents and analysis
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [documentAnswers, setDocumentAnswers] = useState([]);
  const [themes, setThemes] = useState([]);
  const [showAnalysis, setShowAnalysis] = useState(false);

  // Add new state for document upload
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);

  // Add new state for document viewer
  const [showDocuments, setShowDocuments] = useState(false);

  // Add function to fetch documents
  const fetchDocuments = useCallback(async () => {
    try {
      const sessionId = localStorage.getItem("healthcare_session_id");
      if (!sessionId) {
        console.log("No session ID found, skipping document fetch");
        return;
      }
      console.log("Fetching documents with session ID:", sessionId);

      const response = await fetch(
        `https://chatbot-theme-identifier-kzpk.onrender.com/documents?session_id=${sessionId}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch documents");
      }
      const data = await response.json();
      console.log("Fetched documents:", data);
      setUploadedDocuments(data.documents || []);
    } catch (error) {
      console.error("Error fetching documents:", error);
    }
  }, []); // Empty dependency array since we only need sessionId from localStorage

  // Fetch documents only when needed
  useEffect(() => {
    // Only fetch if we have a session ID and showDocuments is true
    const sessionId = localStorage.getItem("healthcare_session_id");
    if (sessionId && showDocuments) {
      fetchDocuments();
    }
  }, [showDocuments, fetchDocuments]); // Only depend on showDocuments state

  // Add function to handle document upload
  const handleDocumentUpload = async (result) => {
    try {
      // Store the session ID from the upload response
      if (result.session_id) {
        localStorage.setItem("healthcare_session_id", result.session_id);
        console.log("Stored session ID:", result.session_id);
      }

      // Add system message about successful upload
      setChatHistory((prev) => [
        ...prev,
        {
          role: "system",
          text: `Successfully uploaded ${
            result.message.split("Successfully uploaded ")[1]
          }. You can now ask questions about the uploaded document.`,
        },
      ]);

      // Show the document viewer and fetch documents
      setShowDocuments(true);
      await fetchDocuments();
    } catch (error) {
      console.error("Error handling document upload:", error);
      setChatHistory((prev) => [
        ...prev,
        {
          role: "system",
          text: "Error processing uploaded document. Please try again.",
          isError: true,
        },
      ]);
    }
  };

  // Modify handleSendMessage to reset input immediately after sending
  const handleSendMessage = async (text = input) => {
    if (!text.trim() || streaming) return;

    // Track user question
    trackUserAction("question_asked", { question: text });

    // Add user message to chat
    setChatHistory((prev) => [...prev, { role: "user", text }]);
    setStreaming(true);
    // Reset input immediately
    setInput("");

    try {
      // Get session ID from localStorage or generate a new one
      const sessionId = localStorage.getItem("healthcare_session_id") || "";

      // Send message to backend
      const response = await fetch(
        "https://chatbot-theme-identifier-kzpk.onrender.com/query",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: text,
            session_id: sessionId,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to process your request");
      }

      const data = await response.json();

      // Store the session ID if it's returned from the server
      if (data.session_id) {
        localStorage.setItem("healthcare_session_id", data.session_id);
      }

      // Update document answers
      setDocumentAnswers(data.individual_answers);

      // Update themes
      setThemes(data.themes || []);

      // Add assistant response to chat
      setChatHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          documentAnswers: data.individual_answers,
          themes: data.themes,
        },
      ]);

      setShowAnalysis(true);
    } catch (error) {
      console.error("Error processing message:", error);
      setChatHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          text:
            error.message ===
            "No documents available for analysis. Please upload documents first."
              ? "Please upload a document first before asking questions. Click the 'Upload' button in the top right corner to add a document."
              : "Sorry, I encountered an error processing your request.",
          isError: true,
        },
      ]);
    } finally {
      setStreaming(false);
    }
  };

  // Handle Enter key press
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div
      id="chatbot"
      className="chat-widget fullscreen"
      style={{ "--primary-color": cfg.primaryColor }}
    >
      <div className="chat-wrapper">
        {/* Header */}
        <div className="chat-header">
          <img
            src={cfg.companyLogo}
            alt={`${cfg.companyName} logo`}
            className="chat-logo"
          />
          <h2 className="chat-title">Chat theme identifier AI Assistant</h2>
          <div className="header-buttons">
            {/* Add View Documents button */}
            <button
              onClick={() => setShowDocuments(!showDocuments)}
              className="view-documents-button"
              aria-label="View documents"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              {showDocuments ? "Hide Documents" : "View Documents"}
            </button>
            {/* Add Upload Documents button */}
            <button
              onClick={() => setShowUploadModal(true)}
              className="upload-documents-button"
              aria-label="Upload documents"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload
            </button>
            <button
              onClick={() => window.closeChatbot?.()}
              className="close-button"
              aria-label="Close chat"
            >
              ×
            </button>
          </div>
        </div>

        {/* Connection status indicator */}
        {connectionStatus !== "CONNECTED" && (
          <div
            className={`connection-status ${connectionStatus.toLowerCase()}`}
          >
            {connectionStatus === "CONNECTING"
              ? "Connecting..."
              : "Disconnected - Please check your connection"}
          </div>
        )}

        {/* Document Viewer */}
        {showDocuments && (
          <div className="document-viewer-container">
            <DocumentViewer
              onDocumentSelect={(doc) => {
                setChatHistory((prev) => [
                  ...prev,
                  {
                    role: "system",
                    text: `Selected document: ${doc.title}. You can now ask questions about this document.`,
                  },
                ]);
              }}
            />
          </div>
        )}

        {/* Chat Content */}
        <div className="chat-content">
          {/* Chat Messages */}
          {chatHistory.map((msg, i) => (
            <div
              key={i}
              className={`chat-block ${msg.role} ${msg.isError ? "error" : ""}`}
            >
              {msg.role !== "system" && (
                <div className="message-label">
                  {msg.role === "user"
                    ? "You"
                    : `${cfg.companyName} AI Assistant`}
                </div>
              )}
              <div
                className={`message ${
                  msg.role === "assistant" ? "assistant-message" : ""
                }`}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.text}
                </ReactMarkdown>
              </div>

              {/* Add document answers and theme analysis after assistant messages */}
              {msg.role === "assistant" && msg.documentAnswers && (
                <DocumentAnswers answers={msg.documentAnswers} />
              )}

              {msg.role === "assistant" && msg.themes && (
                <ThemeAnalysis themes={msg.themes} />
              )}
            </div>
          ))}

          {streaming && (
            <div className="chat-block assistant">
              <div className="message-label">
                {cfg.companyName} AI Assistant
              </div>
              <div className="message">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="chat-input-area">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={cfg.inputPlaceholder}
            rows="1"
            className="chat-input"
            disabled={streaming}
            style={{ height: "55px" }}
          />
          <button
            className="send-button"
            onClick={() => {
              if (input.trim() && !streaming) {
                handleSendMessage();
              }
            }}
            disabled={!input.trim() || streaming}
            aria-label="Send message"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              width="24"
              height="24"
            >
              <path
                fill={streaming ? "#d7d7d7" : "#ffffff"}
                d="M22,11.7V12h-0.1c-0.1,1-17.7,9.5-18.8,9.1c-1.1-0.4,2.4-6.7,3-7.5C6.8,12.9,17.1,12,17.1,12H17c0,0,0-0.2,0-0.2c0,0,0,0,0,0c0-0.4-10.2-1-10.8-1.7c-0.6-0.7-4-7.1-3-7.5C4.3,2.1,22,10.5,22,11.7z"
              ></path>
            </svg>
          </button>
        </div>
      </div>

      {/* Document Upload Modal */}
      <DocumentUploadModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onUpload={handleDocumentUpload}
      />
    </div>
  );
};

export default ChatWidget;
