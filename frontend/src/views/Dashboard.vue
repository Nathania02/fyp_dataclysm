<template>
  <div class="container">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
      <h2>Model Runs</h2>
      <router-link 
        to="/new-run" 
        class="btn btn-primary"
      >
        Run New Model
      </router-link>
    </div>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="runs.length === 0" class="card">
      <p style="text-align: center; color: #7f8c8d;">
        No model runs yet. 
        <router-link to="/new-run" v-if="authStore.isDataScientist">
          Start your first run
        </router-link>
      </p>
    </div>
    
    <div v-else class="card-grid">
      <div v-for="run in runs" :key="run.id" class="card" @click="goToRun(run.id)" style="cursor: pointer;">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
          <h3 style="margin: 0;">Run #{{ run.id }}</h3>
          <span class="status-badge" :class="run.status">{{ run.status }}</span>
        </div>
        
        <div style="color: #7f8c8d; font-size: 0.9rem;">
          <p><strong>Model:</strong> {{ formatModelType(run.model_type) }}</p>
          <p><strong>Dataset:</strong> {{ run.dataset_filename }}</p>
          <p v-if="run.optimal_clusters"><strong>Optimal Clusters:</strong> {{ run.optimal_clusters }}</p>
          <p><strong>Created:</strong> {{ formatDate(run.created_at) }}</p>
          <p v-if="run.completed_at"><strong>Completed:</strong> {{ formatDate(run.completed_at) }}</p>
        </div>
        
        <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
          <span v-if="run.sent_to_clinician" class="status-badge completed">
            Sent to Clinician
          </span>
          <span v-if="run.feedback_added" class="status-badge completed">
            Feedback Added
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import api from '../api'

const router = useRouter()
const authStore = useAuthStore()
const runs = ref([])
const loading = ref(true)

const fetchRuns = async () => {
  try {
    loading.value = true
    const response = await api.getRuns()
    runs.value = response.data
  } catch (error) {
    console.error('Failed to fetch runs:', error)
  } finally {
    loading.value = false
  }
}

const goToRun = (id) => {
  router.push(`/runs/${id}`)
}

const formatModelType = (type) => {
  const types = {
    'kmeans': 'K-Means',
    'kmeans_dtw': 'K-Means with DTW',
    'lca': 'Latent Class Analysis'
  }
  return types[type] || type
}

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString()
}

onMounted(() => {
  fetchRuns()
  // Auto-refresh every 10 seconds
  setInterval(fetchRuns, 10000)
})
</script>