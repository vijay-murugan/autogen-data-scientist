import { useState, useEffect, useRef } from 'react';
import './App.css';

interface Message {
  source: string;
  content: string;
  type: string;
  timestamp: string;
}

interface DatasetFile {
  id: string;
  name: string;
  relative_path: string;
  size_bytes: number;
  file_type: string;
}

function App() {
  const [task, setTask] = useState('');
  const [mode, setMode] = useState<'multi' | 'single' | 'qa'>('single');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRunning, setIsRunning] = useState(false);
<<<<<<< Updated upstream
  const [isLookingUp, setIsLookingUp] = useState(false);
  const [, setArtifacts] = useState<string[]>([]);
  const [datasetInput, setDatasetInput] = useState('');
  const [datasetRef, setDatasetRef] = useState('');
  const [datasetFiles, setDatasetFiles] = useState<DatasetFile[]>([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [error, setError] = useState('');
=======
  const [artifacts, setArtifacts] = useState<string[]>([]);
>>>>>>> Stashed changes

  const workflowEndRef = useRef<HTMLDivElement>(null);
  const dialogueEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    workflowEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    dialogueEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const parseErrorMessage = async (response: Response) => {
    try {
      const payload = await response.json();
      return payload?.detail || payload?.message || `Request failed (${response.status})`;
    } catch {
      return `Request failed (${response.status})`;
    }
  };

  const lookupDataset = async () => {
    if (!datasetInput.trim()) {
      setError('Enter a Kaggle dataset URL or owner/dataset reference.');
      return;
    }

    setError('');
    setIsLookingUp(true);
    setDatasetFiles([]);
    setSelectedFile('');

    try {
      const response = await fetch('http://localhost:8000/api/datasets/lookup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_ref: datasetInput.trim() }),
      });

      if (!response.ok) {
        setError(await parseErrorMessage(response));
        return;
      }

      const payload = await response.json();
      const files = (payload.files || []) as DatasetFile[];
      setDatasetRef(payload.dataset_ref || '');
      setDatasetFiles(files);
      if (files.length > 0) {
        setSelectedFile(files[0].id);
      }
    } catch {
      setError('Failed to contact backend for dataset lookup.');
    } finally {
      setIsLookingUp(false);
    }
  };

  const runTask = async () => {
    if (!task.trim()) return;
    if (!datasetRef || !selectedFile) {
      setError('Lookup a Kaggle dataset and choose a file before running.');
      return;
    }

    setError('');
    setIsRunning(true);
    setMessages([]);
    setArtifacts([]);

    const endpoint = mode === 'qa' ? '/api/qa' : '/api/run';
    const payload =
      mode === 'qa'
        ? { question: task, dataset_ref: datasetRef, selected_file: selectedFile }
        : { task, mode, dataset_ref: datasetRef, selected_file: selectedFile };

    try {
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        setError(await parseErrorMessage(response));
        setIsRunning(false);
        return;
      }

      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '');
            if (dataStr === '[DONE]') {
              setIsRunning(false);
              // Small delay to ensure images are saved
              setTimeout(refreshArtifacts, 1000);
              continue;
            }
            try {
              const msg = JSON.parse(dataStr);
              setMessages(prev => [...prev, msg]);
            } catch (e) {
              console.error('Error parsing JSON:', dataStr);
            }
          }
        }
      }
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Run request failed. Check backend logs and inputs.');
      setIsRunning(false);
    }
  };

  const refreshArtifacts = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/artifacts');
      const data = await response.json();
      setArtifacts(Array.isArray(data.files) ? data.files : []);
    } catch (e) {
      console.error('Artifacts fetch failed:', e);
    }
  };

  const getSourceColor = (source: string) => {
    switch (source.toLowerCase()) {
      case 'user': return '#5e5be5';
      case 'planner': return '#e5a55b';
      case 'datascientist': return '#5be595';
      case 'reviewer': return '#e55b5b';
      case 'dataconsultant': return '#8a5be5';
      case 'analyst': return '#5be5e5';
      default: return '#888';
    }
  };

<<<<<<< Updated upstream
  const dialogueSources = ['user', 'analyst', 'reviewer'];
  if (mode === 'multi') dialogueSources.push('final_result');
  if (mode === 'qa') dialogueSources.push('dataconsultant');
=======
  // Baseline (single agent): show the task on the left, Analyst/tool stream in the workflow column.
  let dialogueMessages: Message[];
  let workflowMessages: Message[];
>>>>>>> Stashed changes

  if (mode === 'single') {
    dialogueMessages = messages.filter(msg => {
      const s = msg.source.toLowerCase();
      return s === 'user' || s === 'error';
    });
    workflowMessages = messages.filter(msg => {
      const s = msg.source.toLowerCase();
      return s !== 'user' && s !== 'error';
    });
  } else {
    const dialogueSources = ['user', 'analyst', 'reviewer'];
    if (mode === 'qa') dialogueSources.push('dataconsultant');
    dialogueMessages = messages.filter(
      msg =>
        dialogueSources.includes(msg.source.toLowerCase()) ||
        msg.source.toLowerCase() === 'error'
    );
    workflowMessages = messages.filter(
      msg =>
        !dialogueSources.includes(msg.source.toLowerCase()) &&
        msg.source.toLowerCase() !== 'error'
    );
  }

  return (
    <div className="dashboard-container">
      <header className="glass-header">
        <h1>
          Multi-Agent Data Analytics <span>v2.0</span>
          {mode === 'single' && (
            <span className="mode-pill">Baseline · single Analyst</span>
          )}
        </h1>
        <div className="status-badge">
          <div className={`dot ${isRunning ? 'active' : ''}`} />
          {isRunning
            ? mode === 'single'
              ? 'Baseline running...'
              : 'Agents Collaborating...'
            : 'Ready'}
        </div>
      </header>

      {/* Panel 1: Question Response */}
      <aside className="dialogue-panel">
        <div className="message-list">
          {dialogueMessages.length === 0 && (
            <div className="placeholder">
              <h3>Interaction</h3>
              <p>
                {mode === 'single'
                  ? 'Your task (and any errors) appear here. The Analyst’s steps stream in the center panel.'
                  : 'Your conversation with the analytics team will appear here.'}
              </p>
            </div>
          )}
          {dialogueMessages.map((msg, i) => (
            <div key={i} className={`message-bubble ${msg.source.toLowerCase()}`}>
              <div className="msg-meta">
                <span className="source" style={{ color: getSourceColor(msg.source) }}>{msg.source}</span>
                <span className="time">{new Date().toLocaleTimeString()}</span>
              </div>
              <div className="msg-content">{msg.content}</div>
            </div>
          ))}
          <div ref={dialogueEndRef} />
        </div>

        <div className="input-panel glass-input">
          <div className="config-row">
            <button 
              className={mode === 'multi' ? 'active' : ''} 
              onClick={() => setMode('multi')}
            >Team Flow</button>
            <button 
              className={mode === 'single' ? 'active' : ''} 
              onClick={() => setMode('single')}
            >Baseline</button>
            <button 
              className={mode === 'qa' ? 'active' : ''} 
              onClick={() => setMode('qa')}
            >Dataset Q&A</button>
          </div>
          <div className="dataset-row">
            <input
              value={datasetInput}
              onChange={(e) => {
                setDatasetInput(e.target.value);
                setDatasetRef('');
                setDatasetFiles([]);
                setSelectedFile('');
              }}
              placeholder="Kaggle URL or owner/dataset"
              disabled={isRunning || isLookingUp}
            />
            <button
              onClick={lookupDataset}
              disabled={isRunning || isLookingUp || !datasetInput.trim()}
            >
              {isLookingUp ? 'Looking up...' : 'Lookup Dataset'}
            </button>
          </div>
          <div className="dataset-meta">
            {datasetRef ? `Resolved: ${datasetRef}` : 'No dataset selected.'}
          </div>
          <div className="dataset-row">
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              disabled={isRunning || isLookingUp || datasetFiles.length === 0}
            >
              {datasetFiles.length === 0 ? (
                <option value="">No files available</option>
              ) : (
                datasetFiles.map((file) => (
                  <option key={file.id} value={file.id}>
                    {file.relative_path} ({file.file_type}, {(file.size_bytes / 1024).toFixed(1)} KB)
                  </option>
                ))
              )}
            </select>
          </div>
          <div className="input-row">
            <input 
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder={
                mode === 'qa'
                  ? 'Ask about the data...'
                  : mode === 'single'
                    ? 'Baseline: one agent runs the full analysis...'
                    : 'Enter a complex analytics task...'
              }
              onKeyDown={(e) => e.key === 'Enter' && runTask()}
              disabled={isRunning}
            />
            <button onClick={runTask} disabled={isRunning || !task.trim()}>
              {isRunning ? '...' : '▶'}
            </button>
          </div>
          {error && <div className="input-error">{error}</div>}
        </div>
      </aside>

      {/* Panel 2: Workflow */}
      <main className="workflow-panel">
        <div className="message-list">
          {workflowMessages.length === 0 && (
            <div className="placeholder">
              <h3>{mode === 'single' ? 'Baseline execution' : 'Workflow Pipeline'}</h3>
              <p>
                {mode === 'single'
                  ? 'The Analyst agent streams code, tool runs, and reasoning here (no Planner/Reviewer).'
                  : 'Internal agent coordination and execution logs.'}
              </p>
            </div>
          )}
          {workflowMessages.map((msg, i) => (
            <div key={i} className={`message-bubble ${msg.source.toLowerCase()}`}>
              <div className="msg-meta">
                <span className="source" style={{ color: getSourceColor(msg.source) }}>{msg.source}</span>
                <span className="time">{new Date().toLocaleTimeString()}</span>
              </div>
              <div className="msg-content">{msg.content}</div>
            </div>
          ))}
          <div ref={workflowEndRef} />
        </div>
      </main>

      {/* Panel 3: Results */}
      <aside className="results-panel glass-panel">
        <h3>Visualization Results</h3>
        <div className="artifact-grid">
          {artifacts.length === 0 ? (
            <div className="empty-artifacts">No charts generated yet for this session.</div>
          ) : (
            artifacts.map((name) => (
              <a
                key={name}
                className="artifact-thumb"
                href={`http://localhost:8000/artifacts/${encodeURIComponent(name)}`}
                target="_blank"
                rel="noreferrer"
              >
                <img
                  src={`http://localhost:8000/artifacts/${encodeURIComponent(name)}`}
                  alt={name}
                />
                <span className="artifact-name">{name}</span>
              </a>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}

export default App;

