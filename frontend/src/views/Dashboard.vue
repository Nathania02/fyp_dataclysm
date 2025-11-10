<template>
  <div class="container">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
      <h2>Model Runs</h2>
      <div style="display: flex; gap: 1rem;">
        <button 
          v-if="!selectMode" 
          @click="toggleSelectMode" 
          class="btn btn-secondary"
        >
          Select Runs
        </button>
        <template v-else>
          <button @click="compareRuns" class="btn btn-primary" :disabled="selectedRuns.length < 2">
            Compare Runs ({{ selectedRuns.length }})
          </button>
          <button @click="cancelSelectMode" class="btn btn-secondary">
            Cancel
          </button>
        </template>
        <router-link 
          to="/new-run" 
          class="btn btn-primary"
        >
          Run New Model
        </router-link>
      </div>
    </div>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="runs.length === 0" class="card">
      <p style="text-align: center; color: #7f8c8d;">
        No model runs yet. 
        <router-link to="/new-run">
          Start your first run
        </router-link>
      </p>
    </div>
    
    <div v-else class="card-grid">
      <div 
        v-for="run in runs" 
        :key="run.id" 
        class="card"
        :class="{ 'card-selected': selectedRuns.includes(run.id) }"
        @click="selectMode ? toggleRunSelection(run.id) : goToRun(run.id)" 
        style="cursor: pointer; position: relative;"
      >
        <div v-if="selectMode" class="checkbox-overlay">
          <input 
            type="checkbox" 
            :checked="selectedRuns.includes(run.id)"
            @click.stop="toggleRunSelection(run.id)"
            class="run-checkbox"
          />
        </div>
        
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
const selectMode = ref(false)
const selectedRuns = ref([])

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
  if (!selectMode.value) {
    router.push(`/runs/${id}`)
  }
}

const toggleSelectMode = () => {
  selectMode.value = true
  selectedRuns.value = []
}

const cancelSelectMode = () => {
  selectMode.value = false
  selectedRuns.value = []
}

const toggleRunSelection = (runId) => {
  const index = selectedRuns.value.indexOf(runId)
  if (index > -1) {
    selectedRuns.value.splice(index, 1)
  } else {
    selectedRuns.value.push(runId)
  }
}

const compareRuns = () => {
  if (selectedRuns.value.length < 2) {
    alert('Please select at least 2 runs to compare')
    return
  }
  router.push({
    name: 'CompareRuns',
    query: { ids: selectedRuns.value.join(',') }
  })
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

<style scoped>
.card-selected {
  border: 2px solid #3498db;
  background: #e3f2fd;
}

.checkbox-overlay {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 10;
}

.run-checkbox {
  width: 20px;
  height: 20px;
  cursor: pointer;
}
</style>