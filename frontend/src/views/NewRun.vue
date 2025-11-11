<template>
  <div class="container">
    <div class="card" style="max-width: 600px; margin: 2rem auto;">
      <h2 style="margin-bottom: 1.5rem;">Run New Model</h2>
      
      <div v-if="error" class="alert alert-danger">{{ error }}</div>
      <div v-if="success" class="alert alert-success">{{ success }}</div>
      
      <form @submit.prevent="handleSubmit">
        <div class="form-group">
          <label for="dataset">Upload Dataset (duckdb)</label>
          <input
            id="dataset"
            type="file"
            accept=".duckdb"
            class="form-control"
            @change="handleDatasetChange"
            required
          />
        </div>

        <div class="form-group">
          <label for="datasetName">Dataset Name</label>
          <input type="text"
                id="datasetName" 
                v-model="datasetName" 
                class="form-control" 
                placeholder="Enter the dataset name"
                required
                />
        </div>

        <div class="form-group">
          <label for="datasetDetails">Metadata</label>
          <input type="text"
                id="datasetDetails" 
                v-model="datasetDetails" 
                class="form-control" 
                placeholder="Enter details of the data"
                required
                />
        </div>

        <div class="form-group">
          <label for="model">Choose Model</label>
          <select id="model" v-model="modelType" class="form-control" required>
            <option value="">Select a model</option>
            <option value="kmeans">K-Means</option>
            <option value="kmeans_dtw">K-Means with DTW for Temporal</option>
            <option value="lca">Latent Class Analysis (LCA)</option>
          </select>
        </div>

        <div class="form-group">
          <label for="parameters">Upload config file</label>
          <input
            id="parameters"
            type="file"
            accept=".yaml"
            class="form-control"
            @change="handleParamsChange"
            required
          />
        </div>
        
        <div style="display: flex; gap: 1rem;">
          <button type="submit" class="btn btn-success" :disabled="submitting" style="flex: 1;">
            {{ submitting ? 'Starting Training...' : 'Start Training' }}
          </button>
          <router-link to="/" class="btn btn-secondary">Cancel</router-link>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'

const router = useRouter()
const modelType = ref('')
const datasetName = ref('')
const datasetDetails = ref('')
const datasetFile = ref(null)
const parameterFile = ref(null)
const error = ref('')
const success = ref('')
const submitting = ref(false)

const handleDatasetChange = (event) => {
  datasetFile.value = event.target.files[0]
}

const handleParamsChange = (event) => {
  parameterFile.value = event.target.files[0]
}

const handleSubmit = async () => {
  try {
    error.value = ''
    success.value = ''
    submitting.value = true
    
    if (!datasetFile.value || !parameterFile.value) {
      error.value = 'Please select a dataset and a configuration file'
      submitting.value = false // Make sure to stop submitting
      return
    }
    
    const response = await api.createRun(
      modelType.value, 
      datasetName.value, 
      datasetDetails.value,
      datasetFile.value,
      parameterFile.value
    )
    success.value = 'Model training started successfully!'
    
    setTimeout(() => {
      router.push(`/runs/${response.data.id}`)
    }, 1500)
  } catch (err) {
    error.value = err.response?.data?.detail || 'Failed to start training'
  } finally {
    submitting.value = false
  }
}
</script>