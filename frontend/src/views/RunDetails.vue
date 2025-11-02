<template>
  <div class="container">
    <div style="margin-bottom: 2rem;">
      <router-link to="/" class="btn btn-secondary">← Back to Dashboard</router-link>
    </div>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="run">
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
          <div>
            <h2>Run #{{ run.id }}</h2>
            <p style="color: #7f8c8d; margin-top: 0.5rem;">
              {{ formatModelType(run.model_type) }}
            </p>
          </div>
          <span class="status-badge" :class="run.status">{{ run.status }}</span>
        </div>
        
        <div style="margin-top: 1.5rem;">
          <p><strong>Dataset:</strong> {{ run.dataset_filename }}</p>
          <p><strong>Created:</strong> {{ formatDate(run.created_at) }}</p>
          <p v-if="run.completed_at"><strong>Completed:</strong> {{ formatDate(run.completed_at) }}</p>
          <p v-if="run.optimal_clusters"><strong>Optimal Clusters:</strong> {{ run.optimal_clusters }}</p>
        </div>
      </div>
      
      <!-- Running Status -->
      <div v-if="run.status === 'running' || run.status === 'pending'" class="card">
        <div class="alert alert-info">
          <h3 style="margin-top: 0;">Model Still Running</h3>
          <p>Your model is being trained. This page will auto-refresh. Come back later to see the results.</p>
          <div class="loading">
            <div class="spinner"></div>
          </div>
        </div>
      </div>
      
      <!-- Results -->
      <div v-else-if="run.status === 'completed'">
        <!-- Plots -->
        <div class="card">
          <h3>Results</h3>
          <p v-if="run.optimal_clusters" style="font-size: 1.1rem; margin: 1rem 0;">
            <strong>Optimal Number of Clusters:</strong> {{ run.optimal_clusters }}
          </p>
          
          <div v-if="plots.length > 0" class="plots-grid">
            <div v-for="plot in plots" :key="plot" class="plot-container">
              <img :src="getPlotUrl(plot)" :alt="plot" />
              <h3>{{ formatPlotName(plot) }}</h3>
            </div>
          </div>
          <div v-else>
            <p style="color: #7f8c8d;">No plots available</p>
          </div>
        </div>
        
        <!-- Notes and Feedback -->
        <div class="card">
          <h3>Notes and Feedback</h3>
          
          <div v-if="notes" style="margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 6px; white-space: pre-wrap; font-family: monospace; font-size: 0.9rem;">
            {{ notes }}
          </div>
          
          <div class="form-group">
            <label>{{ authStore.isClinician ? 'Add Feedback' : 'Add Note' }}</label>
            <textarea
              v-model="newNote"
              class="form-control"
              placeholder="Type your notes or feedback here..."
            ></textarea>
          </div>
          
          <button @click="saveNote" class="btn btn-success">
            {{ authStore.isClinician ? 'Save Feedback' : 'Save Note' }}
          </button>
          
          <div v-if="saveMessage" class="alert alert-success" style="margin-top: 1rem;">
            {{ saveMessage }}
          </div>
        </div>
        
        <!-- Send to Clinician (Data Scientist only) -->
        <div v-if="authStore.isDataScientist && !run.sent_to_clinician" class="card">
          <button @click="sendToClinician" class="btn btn-primary">
            Send Results to Clinician
          </button>
        </div>
        
        <div v-if="run.sent_to_clinician" class="card">
          <div class="alert alert-info">
            Results have been sent to clinicians for feedback.
          </div>
        </div>
      </div>
      
      <!-- Failed Status -->
      <div v-else-if="run.status === 'failed'" class="card">
        <div class="alert alert-danger">
          <h3 style="margin-top: 0;">Training Failed</h3>
          <p>The model training encountered an error. Please try again.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import api from '../api'

const route = useRoute()
const authStore = useAuthStore()
const run = ref(null)
const plots = ref([])
const notes = ref('')
const newNote = ref('')
const saveMessage = ref('')
const loading = ref(true)
let refreshInterval = null

const fetchRun = async () => {
  try {
    const response = await api.getRun(route.params.id)
    run.value = response.data
    
    if (run.value.status === 'completed') {
      await fetchPlots()
      await fetchNotes()
    }
  } catch (error) {
    console.error('Failed to fetch run:', error)
  } finally {
    loading.value = false
  }
}

const fetchPlots = async () => {
  try {
    const response = await api.getRunPlots(route.params.id)
    plots.value = response.data.plots
  } catch (error) {
    console.error('Failed to fetch plots:', error)
  }
}

const fetchNotes = async () => {
  try {
    const response = await api.getNotes(route.params.id)
    notes.value = response.data.content
  } catch (error) {
    console.error('Failed to fetch notes:', error)
  }
}

const saveNote = async () => {
  try {
    saveMessage.value = ''
    if (authStore.isClinician) {
      await api.addFeedback(route.params.id, newNote.value)
      saveMessage.value = 'Feedback saved successfully!'
    } else {
      await api.addNote(route.params.id, newNote.value)
      saveMessage.value = 'Note saved successfully!'
    }
    newNote.value = ''
    await fetchNotes()
    setTimeout(() => saveMessage.value = '', 3000)
  } catch (error) {
    console.error('Failed to save note:', error)
  }
}

const sendToClinician = async () => {
  try {
    await api.sendToClinician(route.params.id)
    await fetchRun()
    alert('Results sent to clinicians successfully!')
  } catch (error) {
    console.error('Failed to send to clinician:', error)
    alert('Failed to send results')
  }
}

const getPlotUrl = (filename) => {
  return api.getPlotFile(route.params.id, filename)
}

const formatModelType = (type) => {
  const types = {
    'kmeans': 'K-Means',
    'kmeans_dtw': 'K-Means with DTW',
    'lca': 'Latent Class Analysis'
  }
  return types[type] || type
}

const formatPlotName = (filename) => {
  return filename
    .replace('.png', '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase())
}

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString()
}

onMounted(() => {
  fetchRun()
  // Auto-refresh every 5 seconds if running
  refreshInterval = setInterval(() => {
    if (run.value && (run.value.status === 'running' || run.value.status === 'pending')) {
      fetchRun()
    }
  }, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>