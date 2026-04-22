import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

interface Message {
  source: string;
  content: string;
  type: string;
  timestamp: string;
}

interface Artifact {
  name: string;
  url: string;
  metadata: any | null;
  modified_at_ms?: number;
}

interface Verdict {
  status: 'PASS' | 'WARN' | 'FAIL' | 'UNKNOWN' | 'CHECKING';
  details: string;
  log?: string;
}

interface DatasetFile {
  id: string;
  name: string;
  relative_path: string;
  size_bytes: number;
  file_type: string;
}

function App() {
  const API_BASE = 'http://localhost:8000';
  const [task, setTask] = useState('');
  const [mode, setMode] = useState<'multi' | 'single' | 'qa' | 'ml' | 'multi_ml'>('multi');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isLookingUp, setIsLookingUp] = useState(false);
  
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [chartQuestions, setChartQuestions] = useState<Record<string, string>>({});
  const [chartAnswers, setChartAnswers] = useState<Record<string, string>>({});
  const [chartLoading, setChartLoading] = useState<Record<string, boolean>>({});
  const [chartVerdicts, setChartVerdicts] = useState<Record<string, Verdict>>({});
  const [expandedVerdict, setExpandedVerdict] = useState<string | null>(null);
  const [expandedChart, setExpandedChart] = useState<string | null>(null);
  const [datasetInput, setDatasetInput] = useState('');
  const [datasetRef, setDatasetRef] = useState('');
  const [datasetFiles, setDatasetFiles] = useState<DatasetFile[]>([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [error, setError] = useState('');

  const workflowEndRef = useRef<HTMLDivElement>(null);
  const dialogueEndRef = useRef<HTMLDivElement>(null);
  const sessionStartedAtRef = useRef<number>(Date.now());

  const scrollToBottom = () => {
    workflowEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    dialogueEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const artifactUrl = (artifactPath: string) => {
    const encodedPath = artifactPath
      .split('/')
      .map((segment) => encodeURIComponent(segment))
      .join('/');
    return `${API_BASE}/artifacts/${encodedPath}?t=${Date.now()}`;
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpandedChart(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

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
      const response = await fetch(`${API_BASE}/api/datasets/lookup`, {
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

  const refreshArtifacts = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/artifacts`);
      if (!response.ok) return;
      const data = await response.json();
      const list = Array.isArray(data.artifacts)
        ? data.artifacts
        : Array.isArray(data.files)
          ? data.files
          : [];
      const filtered = list.filter((art: Artifact) => {
        if (typeof art.modified_at_ms !== 'number') return true;
        return art.modified_at_ms >= sessionStartedAtRef.current;
      });
      setArtifacts(filtered);
      setChartQuestions({});
      setChartAnswers({});
      setChartLoading({});
      setChartVerdicts({});
      // Auto-trigger verification for each chart that has metadata
      filtered.forEach((art: Artifact) => {
        if (art.metadata) {
          void verifyChart(art.name);
        }
      });
    } catch (err) {
      console.error('Error loading artifacts:', err);
    }
  }, [API_BASE]);

  const runTask = async () => {
    if (!task.trim()) return;

    const needsDataset = mode !== 'ml' && mode !== 'multi_ml';
    if (needsDataset && (!datasetRef || !selectedFile)) {
      setError('Lookup a Kaggle dataset and choose a file before running.');
      return;
    }

    setError('');
    setIsRunning(true);
    setMessages([]);
    setArtifacts([]);

    let endpoint: string;
    let payload: Record<string, unknown>;

    if (mode === 'qa') {
      endpoint = '/api/qa';
      payload = {
        question: task,
        dataset_ref: datasetRef,
        selected_file: selectedFile,
      };
    } else if (mode === 'ml') {
      endpoint = '/api/ml';
      payload = { task, mode: 'ml' };
    } else if (mode === 'multi_ml') {
      endpoint = '/api/multi_ml';
      payload = { task, mode: 'multi_ml' };
    } else {
      endpoint = '/api/run';
      payload = { task, mode, dataset_ref: datasetRef, selected_file: selectedFile };
    }

    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
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
              setTimeout(() => void refreshArtifacts(), 1000);
              continue;
            }
            try {
              const msg = JSON.parse(dataStr);
              setMessages((prev) => [...prev, msg]);
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

  const askChartQuestion = async (chartName: string) => {
    const question = chartQuestions[chartName];
    if (!question || !question.trim()) return;

    setChartLoading(prev => ({ ...prev, [chartName]: true }));
    setChartAnswers(prev => ({ ...prev, [chartName]: '' }));

    try {
      const response = await fetch(`${API_BASE}/api/chart-qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chart_name: chartName, question })
      });
      const data = await response.json();
      setChartAnswers(prev => ({ ...prev, [chartName]: data.answer }));
    } catch (err) {
      setChartAnswers(prev => ({ ...prev, [chartName]: `Error: ${err}` }));
    } finally {
      setChartLoading(prev => ({ ...prev, [chartName]: false }));
    }
  };

  const verifyChart = async (chartName: string) => {
    setChartVerdicts(prev => ({ ...prev, [chartName]: { status: 'CHECKING', details: 'Verifier is re-computing values from the dataset...' } }));
    try {
      const response = await fetch(`${API_BASE}/api/verify-chart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chart_name: chartName })
      });
      const data = await response.json();
      setChartVerdicts(prev => ({ ...prev, [chartName]: data }));
    } catch (err) {
      setChartVerdicts(prev => ({ ...prev, [chartName]: { status: 'UNKNOWN', details: `Network error: ${err}` } }));
    }
  };

  const getSourceColor = (source: string) => {
    switch (source.toLowerCase()) {
      case 'user':
        return '#5e5be5';
      case 'planner':
        return '#e5a55b';
      case 'datascientist':
        return '#5be595';
      case 'reviewer':
        return '#e55b5b';
      case 'dataconsultant':
        return '#8a5be5';
      case 'analyst':
        return '#5be5e5';
      case 'modelselector':
        return '#f39c12';
      case 'mlanalyst':
        return '#2ecc71';
      case 'resultsummarizer':
        return '#9b59b6';
      default:
        return '#888';
    }
  };

  let dialogueMessages: Message[];
  let workflowMessages: Message[];

  if (mode === 'single') {
    dialogueMessages = messages.filter((msg) => {
      const s = msg.source.toLowerCase();
      return s === 'user' || s === 'error';
    });
    workflowMessages = messages.filter((msg) => {
      const s = msg.source.toLowerCase();
      return s !== 'user' && s !== 'error';
    });
  } else {
    const dialogueSources = ['user', 'analyst', 'reviewer'];
    if (mode === 'qa') dialogueSources.push('dataconsultant');
    if (mode === 'multi') dialogueSources.push('final_result');
    if (mode === 'multi_ml') dialogueSources.push('resultsummarizer');

    dialogueMessages = messages.filter(
      (msg) =>
        dialogueSources.includes(msg.source.toLowerCase()) || msg.source.toLowerCase() === 'error'
    );
    workflowMessages = messages.filter(
      (msg) =>
        !dialogueSources.includes(msg.source.toLowerCase()) && msg.source.toLowerCase() !== 'error'
    );
  }

  const taskPlaceholder =
    mode === 'qa'
      ? 'Ask about the data...'
      : mode === 'single'
        ? 'Baseline: one agent runs the full analysis...'
        : mode === 'ml'
          ? 'Enter an ML task (e.g., predict or segment)...'
          : mode === 'multi_ml'
            ? 'Enter full analytics + ML objective...'
            : 'Enter a complex analytics task...';

  return (
    <div className="dashboard-container">
      <header className="glass-header">
        <h1>
          Multi-Agent Data Analytics <span>v2.0</span>
          {mode === 'single' && <span className="mode-pill">Baseline · single Analyst</span>}
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
                <span className="source" style={{ color: getSourceColor(msg.source) }}>
                  {msg.source}
                </span>
                <span className="time">{new Date().toLocaleTimeString()}</span>
              </div>
              <div className="msg-content">{msg.content}</div>
            </div>
          ))}
          <div ref={dialogueEndRef} />
        </div>

        <div className="input-panel glass-input">
          <div className="config-row">
            <button className={mode === 'multi' ? 'active' : ''} onClick={() => setMode('multi')}>
              Team Flow
            </button>
            <button
              className={mode === 'single' ? 'active' : ''}
              onClick={() => setMode('single')}
            >
              Baseline
            </button>
            <button className={mode === 'qa' ? 'active' : ''} onClick={() => setMode('qa')}>
              Dataset Q&A
            </button>
            <button className={mode === 'ml' ? 'active' : ''} onClick={() => setMode('ml')}>
              ML Only
            </button>
            <button
              className={mode === 'multi_ml' ? 'active' : ''}
              onClick={() => setMode('multi_ml')}
            >
              Full + ML
            </button>
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
              onClick={() => void lookupDataset()}
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
                    {file.relative_path} ({file.file_type},{' '}
                    {(file.size_bytes / 1024).toFixed(1)} KB)
                  </option>
                ))
              )}
            </select>
          </div>
          <div className="input-row">
            <input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder={taskPlaceholder}
              onKeyDown={(e) => e.key === 'Enter' && void runTask()}
              disabled={isRunning}
            />
            <button onClick={() => void runTask()} disabled={isRunning || !task.trim()}>
              {isRunning ? '...' : '▶'}
            </button>
          </div>
          {error && <div className="input-error">{error}</div>}
        </div>
      </aside>

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
                <span className="source" style={{ color: getSourceColor(msg.source) }}>
                  {msg.source}
                </span>
                <span className="time">{new Date().toLocaleTimeString()}</span>
              </div>
              <div className="msg-content">{msg.content}</div>
            </div>
          ))}
          <div ref={workflowEndRef} />
        </div>
      </main>

      <aside className="results-panel glass-panel">
        <h3>Visualization Results</h3>
        <div className="artifact-grid">
          {artifacts.length === 0 ? (
            <div className="empty-artifacts">
              No charts generated yet for this session.
            </div>
          ) : (
            artifacts.map((art, i) => (
              <div key={i} className="artifact-item" style={{ marginBottom: '24px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '16px' }}>
                <img
                  src={`${API_BASE}${art.url}`}
                  alt={art.metadata?.title || `Chart ${i + 1}`}
                  onClick={() => setExpandedChart(`${API_BASE}${art.url}`)}
                  style={{ width: '100%', borderRadius: '8px', marginBottom: '8px', cursor: 'zoom-in' }}
                  title="Click to expand"
                />
                {art.metadata?.title && (
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px' }}>{art.metadata.title}</div>
                )}

                {art.metadata && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    {(() => {
                      const v = chartVerdicts[art.name];
                      const status = v?.status || 'UNKNOWN';
                      const colors: Record<string, string> = {
                        PASS: '#22c55e',
                        WARN: '#f59e0b',
                        FAIL: '#ef4444',
                        UNKNOWN: '#6b7280',
                        CHECKING: '#3b82f6',
                      };
                      const labels: Record<string, string> = {
                        PASS: '✓ Verified',
                        WARN: '⚠ Warning',
                        FAIL: '✗ Failed',
                        UNKNOWN: '? Unknown',
                        CHECKING: '⟳ Checking...',
                      };
                      return (
                        <>
                          <button
                            onClick={() => setExpandedVerdict(expandedVerdict === art.name ? null : art.name)}
                            disabled={status === 'CHECKING'}
                            style={{
                              padding: '4px 10px',
                              borderRadius: '12px',
                              background: colors[status],
                              color: '#fff',
                              border: 'none',
                              fontSize: '12px',
                              fontWeight: 600,
                              cursor: 'pointer',
                            }}
                            title="Click to see verification details"
                          >
                            {labels[status]}
                          </button>
                          <button
                            onClick={() => verifyChart(art.name)}
                            disabled={status === 'CHECKING'}
                            style={{
                              padding: '4px 10px',
                              borderRadius: '12px',
                              background: 'rgba(255,255,255,0.1)',
                              color: '#fff',
                              border: '1px solid rgba(255,255,255,0.2)',
                              fontSize: '12px',
                              cursor: status === 'CHECKING' ? 'not-allowed' : 'pointer',
                            }}
                          >
                            Re-verify
                          </button>
                        </>
                      );
                    })()}
                  </div>
                )}

                {expandedVerdict === art.name && chartVerdicts[art.name] && (
                  <div
                    style={{
                      marginBottom: '12px',
                      padding: '10px',
                      background: 'rgba(0,0,0,0.3)',
                      borderRadius: '6px',
                      fontSize: '12px',
                      maxHeight: '200px',
                      overflowY: 'auto',
                      whiteSpace: 'pre-wrap',
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: '6px' }}>Verdict details</div>
                    <div style={{ marginBottom: '8px' }}>{chartVerdicts[art.name].details}</div>
                    {chartVerdicts[art.name].log && (
                      <details>
                        <summary style={{ cursor: 'pointer', color: '#888' }}>Show full verifier log</summary>
                        <pre style={{ marginTop: '6px', fontSize: '11px', color: '#aaa' }}>
                          {chartVerdicts[art.name].log}
                        </pre>
                      </details>
                    )}
                  </div>
                )}

                {art.metadata ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <input
                      type="text"
                      placeholder="Ask a question about this chart..."
                      value={chartQuestions[art.name] || ''}
                      onChange={(e) => setChartQuestions(prev => ({ ...prev, [art.name]: e.target.value }))}
                      onKeyDown={(e) => e.key === 'Enter' && askChartQuestion(art.name)}
                      disabled={chartLoading[art.name]}
                      style={{ padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(0,0,0,0.3)', color: '#fff' }}
                    />
                    <button
                      onClick={() => askChartQuestion(art.name)}
                      disabled={chartLoading[art.name] || !chartQuestions[art.name]}
                      style={{ padding: '6px 12px', borderRadius: '6px', background: '#5e5be5', color: '#fff', border: 'none', cursor: 'pointer' }}
                    >
                      {chartLoading[art.name] ? 'Thinking...' : 'Ask'}
                    </button>
                    {chartAnswers[art.name] && (
                      <div style={{ marginTop: '8px', padding: '10px', background: 'rgba(94,91,229,0.15)', borderRadius: '6px', fontSize: '13px', whiteSpace: 'pre-wrap' }}>
                        {chartAnswers[art.name]}
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ fontSize: '12px', color: '#888', fontStyle: 'italic' }}>
                    Q&A unavailable — no underlying data was saved for this chart.
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </aside>
      {expandedChart && (
        <div
          onClick={() => setExpandedChart(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'zoom-out',
            padding: '40px'
          }}
        >
          <img
            src={expandedChart}
            alt="Expanded chart"
            style={{
              maxWidth: '95%',
              maxHeight: '95%',
              objectFit: 'contain',
              borderRadius: '8px',
              boxShadow: '0 10px 40px rgba(0,0,0,0.5)'
            }}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}

export default App;
