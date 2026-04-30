# Multi-Agent Data Analytics Platform
## Presentation Slides

---

# Slide 1: Multi-Agent Data Analytics Platform
### An autonomous, self-verifying data analysis system with human-readable output

> **Team 14 Project | Intelligent Analytics Pipeline**

---

# Slide 2: 5 Stage Agent Architecture

| Agent Role | Responsibility |
|---|---|
| 🧠 **Planner** | Strategic task decomposition, workflow orchestration |
| 🔬 **Research Agent** | Analysis strategy, visualization recommendations, risk assessment |
| 👨💻 **Data Scientist** | Code implementation, data loading, chart generation |
| ✅ **Code Reviewer** | Runtime validation, dependency checking, quality gates |
| 📝 **Result Summarizer** | Final synthesis, natural language answer generation |

---

# Slide 3: Execution Workflow

```
User Query → Planner → ResearchAgent → DataScientist → CodeReviewer → ResultSummarizer → Clean Answer
                                                          ↓
                                                ❌ Issues → Back to DataScientist
                                                ✅ Approved → Continue
```

---

# Slide 4: Key System Features

✅ **Full Agent Collaboration** - All roles work sequentially with verification gates  
✅ **Zero Code Output** - Users only see natural language final answers  
✅ **Automatic Chart Generation** - Visualizations saved with full metadata  
✅ **Interactive Chart Q&A** - Ask follow-up questions about any generated graph  
✅ **Deterministic Verification** - Every chart is mathematically verified against raw dataset  
✅ **Reproducible Artifacts** - All outputs persisted for auditing  

---

# Slide 5: Technical Design Principles

- **Separation of Concerns**: Each agent has single responsibility
- **Controlled Feedback Loops**: Review → Fix cycles until approval
- **Metadata Injection**: Backend stamps dataset reference into every chart
- **Event Driven UI**: Frontend dynamically loads charts when ready
- **No Raw Code Exposure**: Internal workflow hidden from end users

---

# Slide 6: Visualization Pipeline

1.  **Chart Generation**: DataScientist writes matplotlib/seaborn code
2.  **File Persistence**: `plt.savefig()` + `plt.close()` ensures proper disk writing
3.  **Metadata Sidecar**: JSON file generated with chart data points
4.  **Dataset Stamping**: Backend injects source dataset reference
