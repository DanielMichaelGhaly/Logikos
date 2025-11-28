import axios from 'axios';
import { APIResponse } from '../types/mathviz';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // increase timeout to 120s for heavy local models
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🚀 Making API request to: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API response received from: ${response.config.url}`, response.data);
    return response;
  },
  (error) => {
    console.error('❌ API response error:', error);
    
    if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
      throw new Error('Unable to connect to MathViz server. Please ensure the backend is running.');
    }
    
    if (error.response?.status === 404) {
      throw new Error('API endpoint not found. Please check the server configuration.');
    }
    
    if (error.response?.status >= 500) {
      throw new Error('Server error occurred. Please try again later.');
    }
    
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    
    throw new Error(error.message || 'An unexpected error occurred');
  }
);

export const mathvizAPI = {
  /**
   * Solve a mathematical problem
   */
  async solveProblem(problemText: string): Promise<APIResponse> {
    try {
      const response = await apiClient.post('/solve', {
        problem: problemText.trim(),
      });

      return {
        success: true,
        solution: response.data.solution,
        desmos_url: response.data.desmos_url,
      };
    } catch (error) {
      console.error('Error solving problem:', error);
      throw error;
    }
  },

  /**
   * Get solution with visualization
   */
  async solveProblemWithVisualization(problemText: string): Promise<APIResponse> {
    try {
      const response = await apiClient.post('/solve-with-graph', {
        problem: problemText.trim(),
        include_visualization: true,
      });

      return {
        success: true,
        solution: response.data.solution,
        desmos_url: response.data.interactive_graph?.graph_url || response.data.desmos_url,
      };
    } catch (error) {
      console.error('Error solving problem with visualization:', error);
      throw error;
    }
  },

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  /**
   * Get available problem types
   */
  async getProblemTypes(): Promise<string[]> {
    try {
      const response = await apiClient.get('/problem-types');
      return response.data.types || ['algebraic', 'calculus', 'optimization', 'general'];
    } catch (error) {
      console.error('Error fetching problem types:', error);
      // Return default types if endpoint fails
      return ['algebraic', 'calculus', 'optimization', 'general'];
    }
  },

  /**
   * Test the connection with a simple problem
   */
  async testConnection(): Promise<boolean> {
    try {
      const response = await this.solveProblem('2 + 2');
      return response.success;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  },

  /**
   * AI-first chat endpoint: send a message and receive reply, solution, and graph config
   */
  async chat(message: string): Promise<{
    success: boolean;
    reply_text: string;
    solution?: any;
    desmos_url?: string;
    desmos_config?: {
      expressions: string[];
      xmin?: number; xmax?: number; ymin?: number; ymax?: number;
    };
  }> {
    try {
      const response = await apiClient.post('/chat', {
        message,
        include_steps: true,
        include_reasoning: true,
        include_visualization: true,
      });
      return response.data;
    } catch (error) {
      console.error('Error chatting with AI:', error);
      throw error;
    }
  }
};

export default mathvizAPI;
