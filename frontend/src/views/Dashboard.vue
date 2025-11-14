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

    <!-- Filters Section -->
    <div class="filters-section card" style="margin-bottom: 1.5rem;">
      <div class="filters-grid">
        <div class="filter-group">
          <label>Model Type</label>
          <select v-model="filters.modelType" class="form-control">
            <option value="">All Models</option>
            <option value="kmeans">K-Means</option>
            <option value="kmeans_dtw">K-Means with DTW</option>
            <option value="lca">Latent Class Analysis</option>
            <option value="gbtm">GBTM</option>
          </select>
        </div>

        <div class="filter-group">
          <label>Dataset</label>
          <input 
            v-model="filters.dataset" 
            type="text" 
            placeholder="Filter by dataset name"
            class="form-control"
          />
        </div>

        <div class="filter-group">
          <label>Created Date</label>
          <input 
            v-model="filters.createdDate" 
            type="date" 
            class="form-control"
          />
        </div>

        <div class="filter-group">
          <label>Completed Date</label>
          <input 
            v-model="filters.completedDate" 
            type="date" 
            class="form-control"
          />
        </div>

        <div class="filter-group" style="display: flex; align-items: end;">
          <button @click="clearFilters" class="btn btn-secondary" style="width: 100%;">
            Clear Filters
          </button>
        </div>
      </div>
    </div>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="filteredRuns.length === 0" class="card">
      <p style="text-align: center; color: #7f8c8d;">
        No model runs found matching your filters.
        <button @click="clearFilters" class="btn btn-secondary" style="margin-left: 1rem;">
          Clear Filters
        </button>
      </p>
    </div>
    
    <div v-else class="card-grid">
      <div 
        v-for="run in filteredRuns" 
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
          <p><strong>Created by:</strong> {{ getUserName(run.user_email) }}</p>
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
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import api from '../api'

const router = useRouter()
const authStore = useAuthStore()
const runs = ref([])
const loading = ref(true)
const selectMode = ref(false)
const selectedRuns = ref([])
const filters = ref({
  modelType: '',
  dataset: '',
  createdDate: '',
  completedDate: ''
})

const filteredRuns = computed(() => {
  let result = runs.value

  // Filter by model type
  if (filters.value.modelType) {
    result = result.filter(run => run.model_type === filters.value.modelType)
  }

  // Filter by dataset name
  if (filters.value.dataset) {
    result = result.filter(run => 
      run.dataset_filename.toLowerCase().includes(filters.value.dataset.toLowerCase())
    )
  }

  // Filter by created date
  if (filters.value.createdDate) {
    result = result.filter(run => {
      const runDate = new Date(run.created_at).toISOString().split('T')[0]
      return runDate === filters.value.createdDate
    })
  }

  // Filter by completed date
  if (filters.value.completedDate) {
    result = result.filter(run => {
      if (!run.completed_at) return false
      const runDate = new Date(run.completed_at).toISOString().split('T')[0]
      return runDate === filters.value.completedDate
    })
  }

  return result
})

const clearFilters = () => {
  filters.value = {
    modelType: '',
    dataset: '',
    createdDate: '',
    completedDate: ''
  }
}

const getUserName = (email) => {
  if (!email) return 'Unknown'
  return email.split('@')[0]
}

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
    'lca': 'Latent Class Analysis',
    'gbtm': 'GBTM'
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

.filters-section {
  padding: 1.5rem;
}

.filters-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.filter-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #34495e;
  font-size: 0.9rem;
}
</style>