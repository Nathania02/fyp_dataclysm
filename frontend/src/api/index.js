import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor to add token
api.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => Promise.reject(error)
)

// Response interceptor for error handling
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default {
  // Auth
  signup(data) {
    return api.post('/auth/signup', data)
  },
  login(data) {
    return api.post('/auth/login', data)
  },
  getMe() {
    return api.get('/auth/me')
  },

  // Runs
  createRun(modelType, datasetName, datasetDetails, datasetFile, parametersFile) {
    const formData = new FormData()
    formData.append('dataset_file', datasetFile)
    formData.append('parameters_file', parametersFile)

    const modelData = {
      model_type: modelType,
      dataset_name: datasetName,
      dataset_details: datasetDetails
    }
    formData.append('model_data', JSON.stringify(modelData))    
    return api.post(`/runs`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },
  getRuns() {
    return api.get('/runs')
  },
  getRun(runId) {
    return api.get(`/runs/${runId}`)
  },
  getRunPlots(runId) {
    return api.get(`/runs/${runId}/plots`)
  },
  getPlotFile(runId, filename) {
    return `http://localhost:8000/api/runs/${runId}/plots/${filename}?token=${localStorage.getItem('token')}`
  },
  getNotes(runId) {
    return api.get(`/runs/${runId}/notes`)
  },
  addNote(runId, note) {
    return api.post(`/runs/${runId}/notes`, { note })
  },
  addFeedback(runId, feedback) {
    return api.post(`/runs/${runId}/feedback`, { feedback })
  },
  sendToClinician(runId) {
    return api.post(`/runs/${runId}/send-to-clinician`)
  },

  // Notifications
  getNotifications() {
    return api.get('/notifications')
  },
  markNotificationRead(notificationId) {
    return api.put(`/notifications/${notificationId}/read`)
  }
}