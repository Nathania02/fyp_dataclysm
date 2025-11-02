import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const token = ref(localStorage.getItem('token'))

  const isAuthenticated = computed(() => !!token.value)
  const isDataScientist = computed(() => user.value?.role === 'data_scientist')
  const isClinician = computed(() => user.value?.role === 'clinician')

  async function login(email, password) {
    const response = await api.login({ email, password })
    token.value = response.data.access_token
    localStorage.setItem('token', token.value)
    await fetchUser()
  }

  async function signup(email, password, role) {
    await api.signup({ email, password, role })
    await login(email, password)
  }

  async function logout() {
    token.value = null
    user.value = null
    localStorage.removeItem('token')
  }

  async function fetchUser() {
    if (token.value) {
      try {
        const response = await api.getMe()
        user.value = response.data
      } catch (error) {
        await logout()
      }
    }
  }

  return {
    user,
    token,
    isAuthenticated,
    isDataScientist,
    isClinician,
    login,
    signup,
    logout,
    fetchUser
  }
})