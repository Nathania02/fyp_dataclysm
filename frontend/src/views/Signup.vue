<template>
  <div class="container">
    <div class="card" style="max-width: 400px; margin: 4rem auto;">
      <h2 style="margin-bottom: 1.5rem; text-align: center;">Sign Up</h2>
      
      <div v-if="error" class="alert alert-danger">{{ error }}</div>
      
      <form @submit.prevent="handleSubmit">
        <div class="form-group">
          <label for="email">Email</label>
          <input
            id="email"
            v-model="email"
            type="email"
            class="form-control"
            required
          />
        </div>
        
        <div class="form-group">
          <label for="password">Password</label>
          <input
            id="password"
            v-model="password"
            type="password"
            class="form-control"
            required
            minlength="6"
          />
        </div>
        
        <div class="form-group">
          <label for="role">Role</label>
          <select id="role" v-model="role" class="form-control" required>
            <option value="">Select a role</option>
            <option value="data_scientist">Data Scientist</option>
            <option value="clinician">Clinician</option>
          </select>
        </div>
        
        <button type="submit" class="btn btn-primary" style="width: 100%;">
          Sign Up
        </button>
      </form>
      
      <p style="text-align: center; margin-top: 1rem;">
        Already have an account? 
        <router-link to="/login">Login</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const email = ref('')
const password = ref('')
const role = ref('')
const error = ref('')

const handleSubmit = async () => {
  try {
    error.value = ''
    await authStore.signup(email.value, password.value, role.value)
    router.push('/')
  } catch (err) {
    error.value = err.response?.data?.detail || 'Signup failed'
  }
}
</script>