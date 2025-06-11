import React, { useState, useEffect } from "react";
import "./DocumentViewer.css";

const DocumentViewer = ({ onDocumentSelect }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDocument, setSelectedDocument] = useState(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      // Get session ID from localStorage
      const sessionId = localStorage.getItem("healthcare_session_id");
      console.log("Fetching documents with session ID:", sessionId);

      const response = await fetch(
        `https://chatbot-theme-identifier-kzpk.onrender.com/documents?session_id=${sessionId || ""}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch documents");
      }
      const data = await response.json();
      console.log("Fetched documents:", data);
      setDocuments(data.documents || []);
    } catch (err) {
      console.error("Error fetching documents:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDocumentSelect = (doc) => {
    setSelectedDocument(doc);
    if (onDocumentSelect) {
      onDocumentSelect(doc);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const getFileIcon = (fileType) => {
    switch (fileType?.toLowerCase()) {
      case "pdf":
        return "📄";
      case "doc":
      case "docx":
        return "📝";
      case "txt":
        return "📋";
      case "jpg":
      case "jpeg":
      case "png":
        return "🖼️";
      default:
        return "📎";
    }
  };

  if (loading) {
    return (
      <div className="document-viewer">
        <div className="loading">Loading documents...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="document-viewer">
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="document-viewer">
      <div className="document-viewer-header">
        <h3>Uploaded Documents</h3>
        <button onClick={fetchDocuments} className="refresh-button">
          ↻ Refresh
        </button>
      </div>

      {documents.length === 0 ? (
        <div className="no-documents">
          No documents uploaded yet. Click the Upload button to add documents.
        </div>
      ) : (
        <div className="document-list">
          {documents.map((doc, index) => (
            <div
              key={index}
              className={`document-item ${
                selectedDocument?.title === doc.title ? "selected" : ""
              }`}
              onClick={() => handleDocumentSelect(doc)}
            >
              <div className="document-icon">{getFileIcon(doc.file_type)}</div>
              <div className="document-info">
                <div className="document-title">{doc.title}</div>
                <div className="document-meta">
                  <span className="file-type">
                    {doc.file_type?.toUpperCase()}
                  </span>
                  <span className="upload-time">
                    {formatDate(doc.upload_time)}
                  </span>
                </div>
                {doc.description && (
                  <div className="document-description">{doc.description}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DocumentViewer;
