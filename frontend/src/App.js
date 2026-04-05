import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
    document.getElementById('file-input').value = '';
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/analyze', formData);
      setResults(response.data);
    } catch (err) {
      setError('Error analyzing image. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    if (severity === 'SEVERE') return '#e53e3e';
    if (severity === 'MODERATE') return '#dd6b20';
    return '#38a169';
  };

  const getPriorityColor = (priority) => {
    if (priority && priority.includes('RED')) return '#e53e3e';
    if (priority && priority.includes('ORANGE')) return '#dd6b20';
    return '#d69e2e';
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Lung Opacity Detection System</h1>
        <p>AI-Powered Chest X-Ray Analysis - 6 Level Diagnostic System</p>
      </header>

      <div className="main-content">
        <div className="upload-section">
          <h2>Upload Chest X-Ray</h2>

          <div
            className="upload-box"
            onClick={() => document.getElementById('file-input').click()}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              id="file-input"
              style={{ display: 'none' }}
            />
            {preview ? (
              <img src={preview} alt="X-Ray Preview" className="preview-image" />
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">🫁</div>
                <p>Click to upload chest X-ray</p>
                <p>Supports PNG, JPG formats</p>
              </div>
            )}
          </div>

          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
          >
            {loading ? 'Analyzing...' : 'Analyze X-Ray'}
          </button>

          {selectedFile && (
            <button className="clear-btn" onClick={handleClear}>
              Clear Image
            </button>
          )}

          {error && <p className="error">{error}</p>}
        </div>

        {results && (
          <div className="results-section">
            <h2>Analysis Results</h2>

            <div className={`result-card ${results.level1.result === 'ABNORMAL' ? 'abnormal' : 'normal'}`}>
              <h3>Level 1 - Screening</h3>
              <p className="result-value">{results.level1.result}</p>
              <p>Confidence: {results.level1.confidence}%</p>
            </div>

            {results.level2 && (
              <div className="result-card">
                <h3>Level 2 - Disease Classification</h3>
                <p className="result-value">{results.level2.disease.toUpperCase().replace('_', ' ')}</p>
                <p>Confidence: {results.level2.confidence}%</p>
                <div className="probabilities">
                  {Object.entries(results.level2.probabilities).map(([disease, prob]) => (
                    <div key={disease} className="prob-bar">
                      <span>{disease.replace('_', ' ')}</span>
                      <div className="bar-container">
                        <div className="bar" style={{ width: `${prob}%` }}></div>
                      </div>
                      <span>{prob}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results.level3 && (
              <div className="result-card">
                <h3>Level 3 - Location Detection</h3>
                <p className="result-value">{results.level3.location}</p>
                <p>Boxes detected: {results.level3.boxes_detected}</p>
                <p>Bilateral: {results.level3.is_bilateral ? 'Yes' : 'No'}</p>

                {results.annotated_image && (
                  <div className="annotated-image-wrapper">
                    <p>Opacity Detection Overlay</p>
                    <img
                      src={`data:image/png;base64,${results.annotated_image}`}
                      alt="Annotated X-Ray with opacity regions"
                      className="annotated-image"
                    />
                  </div>
                )}
              </div>
            )}

            {results.level4 && (
              <div className="result-card">
                <h3>Level 4 - Affected Area</h3>
                <p className="result-value">{results.level4.affected_percentage}%</p>
                <p>of lung affected</p>
              </div>
            )}

            {results.level5 && (
              <div className="result-card" style={{ borderLeftColor: getSeverityColor(results.level5.severity) }}>
                <h3>Level 5 - Severity</h3>
                <p className="result-value" style={{ color: getSeverityColor(results.level5.severity) }}>
                  {results.level5.severity}
                </p>
              </div>
            )}

            {results.level6 && (
              <div className="result-card" style={{ borderLeftColor: getPriorityColor(results.level6.priority) }}>
                <h3>Level 6 - Clinical Recommendation</h3>
                <p className="result-value" style={{ color: getPriorityColor(results.level6.priority) }}>
                  {results.level6.priority}
                </p>
                <p>{results.level6.action}</p>
                <p>Timeline: {results.level6.timeline}</p>
                <p>Triage: {results.level6.triage}</p>
                {results.level6.notes.map((note, i) => (
                  <p key={i} className="note">Note: {note}</p>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
