<template>
  <div id="app">
    <header class="header" v-if="authStore.isAuthenticated">
      <h1>Model Training Platform</h1>
      <div class="header-right">
        <router-link to="/notifications" class="notification-icon">
          🔔
          <span v-if="unreadCount > 0" class="notification-badge">{{ unreadCount }}</span>
        </router-link>
        <div class="user-info">
          <span>{{ authStore.user?.email }}</span>
          <span class="role-badge" :class="authStore.user?.role">
            {{ authStore.user?.role === 'data_scientist' ? 'Data Scientist' : 'Clinician' }}
          </span>
        </div>
        <button @click="handleLogout" class="btn btn-secondary">Logout</button>
      </div>
    </header>
    <router-view />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from './stores/auth'
import api from './api'

const authStore = useAuthStore()
const router = useRouter()
const unreadCount = ref(0)

const handleLogout = async () => {
  await authStore.logout()
  router.push('/login')
}

const fetchNotifications = async () => {
  if (authStore.isAuthenticated) {
    try {
      const response = await api.getNotifications()
      unreadCount.value = response.data.filter(n => !n.is_read).length
    } catch (error) {
      console.error('Failed to fetch notifications:', error)
    }
  }
}

onMounted(() => {
  fetchNotifications()
  // Poll for notifications every 30 seconds
  setInterval(fetchNotifications, 30000)
})
</script>

<style>
.notification-icon {
  position: relative;
  font-size: 1.5rem;
  text-decoration: none;
}

.notification-badge {
  position: absolute;
  top: -5px;
  right: -5px;
}
</style>