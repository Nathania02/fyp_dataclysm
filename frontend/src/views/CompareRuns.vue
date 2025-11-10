<template>
  <div class="container">
    <div style="margin-bottom: 2rem;">
      <router-link to="/" class="btn btn-secondary">← Back to Dashboard</router-link>
    </div>
    
    <h2 style="margin-bottom: 2rem;">Compare Runs</h2>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="runs.length === 0" class="card">
      <p style="text-align: center; color: #7f8c8d;">
        No runs to compare. Please select runs from the dashboard.
      </p>
    </div>
    
    <div v-else>
      <!-- Run Summary Cards -->
      <div class="comparison-header">
        <div 
          v-for="run in runs" 
          :key="run.id" 
          class="card comparison-card"
        >
          <h3>Run #{{ run.id }}</h3>
          <div style="color: #7f8c8d; font-size: 0.9rem;">
            <p><strong>Model:</strong> {{ formatModelType(run.model_type) }}</p>
            <p><strong>Status:</strong> <span class="status-badge" :class="run.status">{{ run.status }}</span></p>
            <p><strong>Dataset:</strong> {{ run.dataset_filename }}</p>
            <p v-if="run.optimal_clusters"><strong>Optimal Clusters:</strong> {{ run.optimal_clusters }}</p>
            <p><strong>Created:</strong> {{ formatDate(run.created_at) }}</p>
            <p v-if="run.completed_at"><strong>Completed:</strong> {{ formatDate(run.completed_at) }}</p>
          </div>
        </div>
      </div>
      
      <!-- Comparison Table -->
      <div class="card" style="overflow-x: auto;">
        <h3>Run Comparison</h3>
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th v-for="run in runs" :key="run.id">Run #{{ run.id }}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Model Type</strong></td>
              <td v-for="run in runs" :key="run.id">{{ formatModelType(run.model_type) }}</td>
            </tr>
            <tr>
              <td><strong>Status</strong></td>
              <td v-for="run in runs" :key="run.id">
                <span class="status-badge" :class="run.status">{{ run.status }}</span>
              </td>
            </tr>
            <tr>
              <td><strong>Optimal Clusters</strong></td>
              <td v-for="run in runs" :key="run.id">{{ run.optimal_clusters || 'N/A' }}</td>
            </tr>
            <tr>
              <td><strong>Dataset</strong></td>
              <td v-for="run in runs" :key="run.id">{{ run.dataset_filename }}</td>
            </tr>
            <tr>
              <td><strong>Duration</strong></td>
              <td v-for="run in runs" :key="run.id">{{ calculateDuration(run) }}</td>
            </tr>
            <tr>
              <td><strong>Sent to Clinician</strong></td>
              <td v-for="run in runs" :key="run.id">{{ run.sent_to_clinician ? 'Yes' : 'No' }}</td>
            </tr>
            <tr>
              <td><strong>Feedback Added</strong></td>
              <td v-for="run in runs" :key="run.id">{{ run.feedback_added ? 'Yes' : 'No' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      
      <!-- Plots Comparison -->
      <div v-for="run in completedRuns" :key="run.id" class="card">
        <h3>Run #{{ run.id }} - {{ formatModelType(run.model_type) }}</h3>
        
        <div v-if="runPlots[run.id] && runPlots[run.id].length > 0" class="plots-grid">
          <div v-for="plot in runPlots[run.id]" :key="plot" class="plot-container">
            <img :src="getPlotUrl(run.id, plot)" :alt="plot" />
            <h4>{{ formatPlotName(plot) }}</h4>
          </div>
        </div>
        <div v-else>
          <p style="color: #7f8c8d;">No plots available for this run</p>
        </div>
      </div>
      
      <!-- Side by Side Comparison (if same model type) -->
      <div v-if="canCompareSideBySide" class="card">
        <h3>Side-by-Side Plot Comparison</h3>
        <p style="color: #7f8c8d; margin-bottom: 1rem;">
          Comparing plots from runs with the same model type
        </p>
        
        <div v-for="plotName in commonPlots" :key="plotName">
          <h4 style="margin-top: 2rem;">{{ formatPlotName(plotName) }}</h4>
          <div class="side-by-side-plots">
            <div v-for="run in completedRuns" :key="run.id" class="side-by-side-plot">
              <h5>Run #{{ run.id }}</h5>
              <img 
                v-if="runPlots[run.id] && runPlots[run.id].includes(plotName)"
                :src="getPlotUrl(run.id, plotName)" 
                :alt="plotName" 
              />
              <p v-else style="color: #7f8c8d;">Plot not available</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import api from '../api'

const route = useRoute()
const authStore = useAuthStore()
const runs = ref([])
const runPlots = ref({})
const loading = ref(true)

const completedRuns = computed(() => {
  return runs.value.filter(run => run.status === 'completed')
})

const canCompareSideBySide = computed(() => {
  if (completedRuns.value.length < 2) return false
  const modelTypes = completedRuns.value.map(r => r.model_type)
  return modelTypes.every(type => type === modelTypes[0])
})

const commonPlots = computed(() => {
  if (completedRuns.value.length === 0) return []
  
  const allPlots = completedRuns.value.map(run => runPlots.value[run.id] || [])
  if (allPlots.length === 0) return []
  
  // Find common plots across all runs
  return allPlots[0].filter(plot => 
    allPlots.every(plots => plots.includes(plot))
  )
})

const fetchRuns = async () => {
  try {
    loading.value = true
    const runIds = route.query.ids.split(',').map(id => parseInt(id))
    
    // Fetch each run's data
    const runPromises = runIds.map(id => api.getRun(id))
    const runResponses = await Promise.all(runPromises)
    runs.value = runResponses.map(res => res.data)
    
    // Fetch plots for completed runs
    for (const run of runs.value) {
      if (run.status === 'completed') {
        try {
          const plotsResponse = await api.getRunPlots(run.id)
          runPlots.value[run.id] = plotsResponse.data.plots
        } catch (error) {
          console.error(`Failed to fetch plots for run ${run.id}:`, error)
          runPlots.value[run.id] = []
        }
      }
    }
  } catch (error) {
    console.error('Failed to fetch runs:', error)
  } finally {
    loading.value = false
  }
}

const getPlotUrl = (runId, filename) => {
  return api.getPlotFile(runId, filename)
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

const calculateDuration = (run) => {
  if (!run.completed_at || !run.created_at) return 'N/A'
  
  const start = new Date(run.created_at)
  const end = new Date(run.completed_at)
  const durationMs = end - start
  const minutes = Math.floor(durationMs / 60000)
  const seconds = Math.floor((durationMs % 60000) / 1000)
  
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`
  }
  return `${seconds}s`
}

onMounted(() => {
  if (!route.query.ids) {
    return
  }
  fetchRuns()
})
</script>

<style scoped>
.comparison-header {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.comparison-card {
  margin: 0;
}

.comparison-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}

.comparison-table th,
.comparison-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid #e0e0e0;
}

.comparison-table th {
  background: #f8f9fa;
  font-weight: 600;
  position: sticky;
  top: 0;
}

.comparison-table tbody tr:hover {
  background: #f8f9fa;
}

.comparison-table td:first-child {
  font-weight: 500;
  background: #fafafa;
}

.plots-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 1.5rem;
  margin-top: 1.5rem;
}

.plot-container {
  text-align: center;
}

.plot-container img {
  max-width: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.plot-container h4 {
  margin-top: 0.75rem;
  font-size: 0.95rem;
  color: #34495e;
}

.side-by-side-plots {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

.side-by-side-plot {
  text-align: center;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.side-by-side-plot h5 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.side-by-side-plot img {
  max-width: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
</style>