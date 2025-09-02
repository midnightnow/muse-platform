import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, 
  Settings, 
  Brain, 
  Code, 
  FileText, 
  Zap, 
  Eye, 
  Activity,
  Users,
  Database,
  Cpu,
  Cloud,
  CheckCircle,
  AlertCircle,
  Clock,
  BarChart3,
  Loader2,
  RefreshCw
} from 'lucide-react';

// Firestore imports for persistent state (optional)
// import { initializeApp } from 'firebase/app';
// import { getFirestore, doc, setDoc, getDoc } from 'firebase/firestore';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const MissionControlDashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeWorkflow, setActiveWorkflow] = useState(null);
  const [selectedModels, setSelectedModels] = useState({});
  const [dashboardData, setDashboardData] = useState({
    workflow_templates: [],
    available_models: [],
    context_packs: [],
    system_status: {
      orchestrator: 'unknown',
      mlx_training: 'unknown',
      vector_db: 'unknown',
      api_models: 'unknown'
    },
    active_workflows: 0,
    training_queue: 0,
    memory_usage: 0,
    total_contexts: 0
  });
  
  const [ws, setWs] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [realtimeUpdates, setRealtimeUpdates] = useState([]);

  // Fetch dashboard data from API
  const fetchDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/dashboard`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setDashboardData(data);
      
      // Restore selected models from localStorage or Firestore
      const savedModels = localStorage.getItem('selectedModels');
      if (savedModels) {
        setSelectedModels(JSON.parse(savedModels));
      }
      
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data. Please check if the API is running.');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initialize WebSocket connection
  const initWebSocket = useCallback(() => {
    const websocket = new WebSocket(`ws://localhost:8000/ws`);
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWsConnected(true);
    };
    
    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      console.log('WebSocket message:', message);
      
      // Handle different message types
      switch (message.type) {
        case 'model_update':
          updateModelStatus(message.model_id, message.status);
          break;
        case 'workflow_progress':
          updateWorkflowProgress(message);
          break;
        case 'metrics_update':
          updateModelMetrics(message);
          break;
        default:
          // Add to realtime updates feed
          setRealtimeUpdates(prev => [message, ...prev.slice(0, 9)]);
      }
    };
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);
      // Attempt to reconnect after 5 seconds
      setTimeout(() => initWebSocket(), 5000);
    };
    
    setWs(websocket);
    
    return websocket;
  }, []);

  // Update model status in real-time
  const updateModelStatus = (modelId, status) => {
    setDashboardData(prev => ({
      ...prev,
      available_models: prev.available_models.map(model =>
        model.id === modelId ? { ...model, status } : model
      )
    }));
  };

  // Update model metrics in real-time
  const updateModelMetrics = (message) => {
    setDashboardData(prev => ({
      ...prev,
      available_models: prev.available_models.map(model =>
        model.id === message.model_id 
          ? { 
              ...model, 
              current_load: message.current_load,
              total_requests: message.total_requests 
            } 
          : model
      )
    }));
  };

  // Update workflow progress
  const updateWorkflowProgress = (message) => {
    // Update UI with workflow progress
    // This could update a progress bar or status indicator
    console.log('Workflow progress:', message);
  };

  // Execute workflow
  const executeWorkflow = async () => {
    if (!activeWorkflow) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/workflows/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          workflow_id: activeWorkflow.id,
          model_preferences: selectedModels
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to execute workflow');
      }
      
      const execution = await response.json();
      console.log('Workflow execution started:', execution);
      
      // Show notification or update UI
      setRealtimeUpdates(prev => [
        { type: 'info', message: `Workflow "${activeWorkflow.name}" started` },
        ...prev.slice(0, 9)
      ]);
      
    } catch (err) {
      console.error('Error executing workflow:', err);
      setError('Failed to execute workflow');
    }
  };

  // Save selected models to localStorage (or Firestore)
  const saveSelectedModels = useCallback(() => {
    localStorage.setItem('selectedModels', JSON.stringify(selectedModels));
    // Could also save to Firestore here
  }, [selectedModels]);

  // Initialize on component mount
  useEffect(() => {
    fetchDashboardData();
    const websocket = initWebSocket();
    
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, [fetchDashboardData, initWebSocket]);

  // Save selected models when they change
  useEffect(() => {
    saveSelectedModels();
  }, [selectedModels, saveSelectedModels]);

  // Component sub-components
  const ModelCard = ({ model, isSelected, onToggle }) => (
    <div 
      className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
        isSelected 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={() => onToggle(model.id)}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold">{model.name}</h4>
        <div className="flex items-center gap-1">
          <div className={`w-2 h-2 rounded-full ${
            model.status === 'online' ? 'bg-green-500 animate-pulse' : 
            model.status === 'training' ? 'bg-yellow-500' : 'bg-red-500'
          }`} />
          <span className="text-xs text-gray-500">{model.type}</span>
        </div>
      </div>
      <div className="text-sm text-gray-600 mb-3">
        <div>Specialty: <span className="font-medium">{model.specialty}</span></div>
        <div className="flex gap-4 mt-1">
          <span>Cost: {model.cost}</span>
          <span>Speed: {model.speed}</span>
        </div>
        {model.current_load !== undefined && (
          <div className="mt-2">
            <div className="flex justify-between text-xs mb-1">
              <span>Load</span>
              <span>{Math.round(model.current_load * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1">
              <div 
                className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                style={{ width: `${model.current_load * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const WorkflowCard = ({ template, isActive, onSelect }) => {
    const Icon = template.icon === 'FileText' ? FileText : 
                 template.icon === 'Code' ? Code : Brain;
    
    return (
      <div 
        className={`p-6 rounded-xl border-2 cursor-pointer transition-all ${
          isActive 
            ? 'border-indigo-500 bg-indigo-50' 
            : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
        }`}
        onClick={() => onSelect(template)}
      >
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 rounded-lg bg-indigo-100">
            <Icon className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h3 className="text-xl font-bold">{template.name}</h3>
            <p className="text-gray-600">{template.description}</p>
          </div>
        </div>
        
        <div className="mb-4">
          <h4 className="font-semibold mb-2">Workflow Steps:</h4>
          <div className="flex gap-2 flex-wrap">
            {template.steps.map((step, idx) => (
              <span key={idx} className="px-3 py-1 bg-gray-100 rounded-full text-sm">
                {step}
              </span>
            ))}
          </div>
        </div>

        <div>
          <h4 className="font-semibold mb-2">Suggested Models:</h4>
          <div className="flex gap-2 flex-wrap">
            {template.suggested_models.map((model, idx) => (
              <span key={idx} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                {model}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-indigo-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading Mission Control...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center bg-white p-8 rounded-lg shadow-lg max-w-md">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-900 mb-2">Connection Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={fetchDashboardData}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2 mx-auto"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Mission Control</h1>
              <p className="text-lg text-gray-600">Master Orchestrator LLM Platform</p>
            </div>
            
            {/* System Status Indicators */}
            <div className="flex gap-4">
              {Object.entries(dashboardData.system_status).map(([service, status]) => (
                <div key={service} className="flex items-center gap-2 px-3 py-2 bg-white rounded-lg shadow-sm">
                  <div className={`w-2 h-2 rounded-full ${
                    status === 'active' ? 'bg-green-500 animate-pulse' : 
                    status === 'unknown' ? 'bg-gray-400' : 'bg-yellow-500'
                  }`} />
                  <span className="text-sm font-medium capitalize">{service.replace('_', ' ')}</span>
                </div>
              ))}
              
              {/* WebSocket Status */}
              <div className="flex items-center gap-2 px-3 py-2 bg-white rounded-lg shadow-sm">
                <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                <span className="text-sm font-medium">WebSocket</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column - Workflow Templates */}
          <div className="xl:col-span-2">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Zap className="w-6 h-6 text-indigo-600" />
                Workflow Templates
              </h2>
              
              <div className="space-y-6">
                {dashboardData.workflow_templates.map(template => (
                  <WorkflowCard 
                    key={template.id} 
                    template={template}
                    isActive={activeWorkflow?.id === template.id}
                    onSelect={setActiveWorkflow}
                  />
                ))}
              </div>

              {/* Custom Workflow Builder */}
              <div className="mt-8 p-6 border-2 border-dashed border-gray-300 rounded-xl text-center">
                <Settings className="w-8 h-8 text-gray-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Custom Workflow</h3>
                <p className="text-gray-500 mb-4">Build your own orchestration pipeline</p>
                <button className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors">
                  Open Workflow Builder
                </button>
              </div>
            </div>

            {/* Real-time Updates Feed */}
            {realtimeUpdates.length > 0 && (
              <div className="mt-6 bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-green-600" />
                  Real-time Updates
                </h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {realtimeUpdates.map((update, idx) => (
                    <div key={idx} className="text-sm text-gray-600 p-2 bg-gray-50 rounded">
                      {update.message || JSON.stringify(update)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Model Selection & Context */}
          <div className="space-y-6">
            {/* Model Selection */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-blue-600" />
                Available Models ({dashboardData.available_models.filter(m => m.status === 'online').length}/{dashboardData.available_models.length})
              </h2>
              
              <div className="space-y-3">
                {dashboardData.available_models.map(model => (
                  <ModelCard 
                    key={model.id}
                    model={model}
                    isSelected={selectedModels[model.id]}
                    onToggle={(id) => setSelectedModels(prev => ({
                      ...prev,
                      [id]: !prev[id]
                    }))}
                  />
                ))}
              </div>
            </div>

            {/* Context Packs */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-green-600" />
                Context Packs
              </h2>
              
              <div className="space-y-3">
                {dashboardData.context_packs.map(pack => (
                  <div key={pack.id} className="p-3 rounded-lg border hover:bg-gray-50 cursor-pointer">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="font-semibold">{pack.name}</h4>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    </div>
                    <div className="text-sm text-gray-600">
                      <div>{pack.type} • {pack.size}</div>
                      <div>Updated: {pack.last_updated}</div>
                    </div>
                  </div>
                ))}
              </div>
              
              <button className="w-full mt-4 px-4 py-2 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-400 hover:text-gray-700 transition-colors">
                + Add Context Pack
              </button>
            </div>

            {/* Quick Stats */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-600" />
                System Stats
              </h2>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Active Workflows</span>
                  <span className="font-bold text-green-600">{dashboardData.active_workflows}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Training Queue</span>
                  <span className="font-bold text-blue-600">{dashboardData.training_queue} tasks</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Memory Usage</span>
                  <span className="font-bold text-yellow-600">{Math.round(dashboardData.memory_usage * 100)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Total Contexts</span>
                  <span className="font-bold text-purple-600">{dashboardData.total_contexts.toFixed(1)}M tokens</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Execute Button */}
        {activeWorkflow && (
          <div className="fixed bottom-8 right-8">
            <button 
              onClick={executeWorkflow}
              className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-2xl shadow-2xl hover:shadow-3xl transition-all text-lg font-bold flex items-center gap-3"
            >
              <Play className="w-6 h-6" />
              Launch {activeWorkflow.name}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default MissionControlDashboard;