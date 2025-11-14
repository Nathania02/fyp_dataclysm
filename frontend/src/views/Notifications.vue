<template>
  <div class="container">
    <div style="margin-bottom: 2rem;">
      <router-link to="/" class="btn btn-secondary">← Back to Dashboard</router-link>
    </div>
    <h2 style="margin-bottom: 2rem;">Notifications</h2>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="notifications.length === 0" class="card">
      <p style="text-align: center; color: #7f8c8d;">
        No notifications yet.
      </p>
    </div>
    
    <div v-else class="card" style="padding: 0;">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        class="notification-item"
        :class="{ unread: !notification.is_read }"
        @click="handleNotificationClick(notification)"
      >
        <div style="display: flex; justify-content: space-between; align-items: start;">
          <div style="flex: 1;">
            <p style="margin: 0; font-weight: 500;">{{ notification.message }}</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #7f8c8d;">
              {{ formatDate(notification.created_at) }}
            </p>
          </div>
          <span v-if="!notification.is_read" style="width: 8px; height: 8px; background: #3498db; border-radius: 50%; margin-top: 0.5rem;"></span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'

const router = useRouter()
const notifications = ref([])
const loading = ref(true)

const fetchNotifications = async () => {
  try {
    loading.value = true
    const response = await api.getNotifications()
    notifications.value = response.data
  } catch (error) {
    console.error('Failed to fetch notifications:', error)
  } finally {
    loading.value = false
  }
}

const handleNotificationClick = async (notification) => {
  try {
    if (!notification.is_read) {
      await api.markNotificationRead(notification.id)
      notification.is_read = true
    }
    router.push(`/runs/${notification.run_id}`)
  } catch (error) {
    console.error('Failed to mark notification as read:', error)
  }
}

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString()
}

onMounted(() => {
  fetchNotifications()
})
</script>