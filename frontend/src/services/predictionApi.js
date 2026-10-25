import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10s timeout
});

export async function checkApiHealth() {
  try {
    const response = await apiClient.get('/api/health');
    return { success: true, data: response.data };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || error.message || 'Unable to connect to backend API server.',
    };
  }
}

export async function predictSubscription(customerPayload) {
  try {
    const response = await apiClient.post('/api/predict', customerPayload);
    return { success: true, data: response.data };
  } catch (error) {
    let errorMessage = 'Failed to obtain prediction from model service.';
    if (error.response) {
      if (error.response.status === 422) {
        // Validation error
        const details = error.response.data?.detail;
        if (Array.isArray(details)) {
          errorMessage = details.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join(', ');
        } else {
          errorMessage = error.response.data?.detail || 'Input validation failed.';
        }
      } else if (error.response.status === 503) {
        errorMessage = 'Model service is currently unavailable. Please verify model file export.';
      } else {
        errorMessage = error.response.data?.detail || `Server error (${error.response.status}).`;
      }
    } else if (error.request) {
      errorMessage = 'Cannot reach backend API at ' + API_BASE_URL + '. Please ensure FastAPI server is running.';
    }
    return { success: false, error: errorMessage };
  }
}

export default apiClient;
