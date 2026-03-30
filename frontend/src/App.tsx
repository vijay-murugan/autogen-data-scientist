import { useState, useEffect, useRef } from 'react';
import './App.css';

interface Message {
  source: string;
  content: string;
  type: string;
  timestamp: string;
}

function App() {
  const [task, setTask] = useState('');
  const [mode, setMode] = useState<'multi' | 'single' | 'qa'>('multi');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [, setArtifacts] = useState<string[]>([]);

  const workflowEndRef = useRef<HTMLDivElement>(null);
  const dialogueEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    workflowEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    dialogueEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const runTask = async () => {
    if (!task) return;
    setIsRunning(true);
    setMessages([]);
    setArtifacts([]);

    const endpoint = mode === 'qa' ? '/api/qa' : '/api/run';
    const payload = mode === 'qa' ? { question: task } : { task, mode };

    try {
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

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
      setIsRunning(false);
    }
  };

  const refreshArtifacts = async () => {
    // Basic polling to find new images
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

  const dialogueSources = ['user', 'analyst', 'reviewer'];
  if (mode === 'qa') dialogueSources.push('dataconsultant');

  const dialogueMessages = messages.filter(msg => 
    dialogueSources.includes(msg.source.toLowerCase()) || msg.source.toLowerCase() === 'error'
  );
  
  const workflowMessages = messages.filter(msg => 
    !dialogueSources.includes(msg.source.toLowerCase()) && msg.source.toLowerCase() !== 'error'
  );

  return (
    <div className="dashboard-container">
      <header className="glass-header">
        <h1>Multi-Agent Data Analytics <span>v2.0</span></h1>
        <div className="status-badge">
          <div className={`dot ${isRunning ? 'active' : ''}`} />
          {isRunning ? 'Agents Collaborating...' : 'Ready'}
        </div>
      </header>

      {/* Panel 1: Question Response */}
      <aside className="dialogue-panel">
        <div className="message-list">
          {dialogueMessages.length === 0 && (
            <div className="placeholder">
              <h3>Interaction</h3>
              <p>Your conversation with the analytics team will appear here.</p>
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
          <div className="input-row">
            <input 
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder={mode === 'qa' ? "Ask about the data..." : "Enter a complex analytics task..."}
              onKeyDown={(e) => e.key === 'Enter' && runTask()}
              disabled={isRunning}
            />
            <button onClick={runTask} disabled={isRunning || !task}>
              {isRunning ? '...' : '▶'}
            </button>
          </div>
        </div>
      </aside>

      {/* Panel 2: Workflow */}
      <main className="workflow-panel">
        <div className="message-list">
          {workflowMessages.length === 0 && (
            <div className="placeholder">
              <h3>Workflow Pipeline</h3>
              <p>Internal agent coordination and execution logs.</p>
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
           <div className="empty-artifacts">
             No charts generated yet for this session.
           </div>
        </div>
      </aside>
    </div>
  );
}

export default App;

